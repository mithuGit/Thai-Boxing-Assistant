# type: ignore
import cv2
import mediapipe as mp
import numpy as np
import csv
import matplotlib.pyplot as plt
from fastdtw import fastdtw
import pandas as pd
from scipy.spatial.distance import euclidean

def get_video_file():
    """Fragt im Terminal den Pfad zum Video ab."""
    video_path = input("Bitte den Pfad zum Video eingeben (z.B. video.mp4): ").strip()
    return video_path

def process_video(video_path):
    """
    Liest das Video frameweise, extrahiert die Pose mit MediaPipe
    und speichert für jeden Frame einen Feature-Vektor (alle 33 Landmarks mit x,y,z,visibility).
    Dabei werden x und y relativ zur Hüftposition (Landmark 23 und 24) normalisiert.
    """
    cap = cv2.VideoCapture(video_path)
    mp_pose = mp.solutions.pose
    
    # --- Automatische Orientierungserkennung ---
    rotate_video = False
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose_detector:
        found_frame = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_detector.process(image_rgb)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # Nase ist Landmark 0, Hüfte sind Landmark 23 und 24
                nose_y = landmarks[0].y
                hip1_y = landmarks[23].y
                hip2_y = landmarks[24].y
                hips_center_y = (hip1_y + hip2_y) / 2.0
                # In einem korrekt orientierten Bild sollte die Nase über dem Hüftzentrum liegen (kleinere y-Werte)
                if nose_y > hips_center_y:
                    rotate_video = True
                    print("Video scheint kopfstehend zu sein. Drehe es um 180° vor der Verarbeitung.")
                else:
                    print("Video-Orientierung ist korrekt.")
                found_frame = True
                break
        # Setze den Video-Stream zurück an den Anfang
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # --- Ende der Orientierungserkennung ---
    
    # Starte die eigentliche Verarbeitung
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    sequence = []
    
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Wende Rotation an, wenn das Video kopfstehend ist
        if rotate_video:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            features = []
            for lm in landmarks:
                features.extend([lm.x, lm.y, lm.z, lm.visibility])
            
            if len(features) >= 132:
                hip1 = np.array(features[23*4:23*4+2])
                hip2 = np.array(features[24*4:24*4+2])
                center = (hip1 + hip2) / 2.0
                for i in range(0, len(features), 4):
                    features[i]   = features[i]   - center[0]
                    features[i+1] = features[i+1] - center[1] 
           
           
            sequence.append(features)
        else:
            sequence.append([0]*132)
        
        frame_index += 1
    
    cap.release()
    sequence = np.array(sequence)
    print(f"Verarbeitete {len(sequence)} Frames.")
    return sequence

def load_reference_sequence(csv_file, normalize=True):
    """
    Lädt eine Referenzsequenz (z. B. aus perfect_punch.csv) und normalisiert sie analog zu process_video.
    Erwartet wird, dass in jeder Zeile (ohne Frame- und Label-Spalte) 132 Werte stehen.
    """
    sequence = []
    expected_length = 132  # 33 Landmarks * 4 Werte
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Überspringe Header
        for row in reader:
            if not row:
                continue
            features = np.array(list(map(float, row[1:-1]))) #entferne Frames und Label
            if len(features) != expected_length:
                print(f"Übersprungen, da Länge {len(features)} nicht passt.")
                continue
            
            if normalize:
                hip1 = features[23*4:23*4+2]
                hip2 = features[24*4:24*4+2]
                center = (hip1 + hip2) / 2
                for i in range(0, len(features), 4):
                    features[i]   -= center[0]
                    features[i+1] -= center[1]
           
            sequence.append(features)
    sequence = np.array(sequence)
    print(f"Referenzsequenz geladen aus '{csv_file}' mit Shape: {sequence.shape}")
    return sequence

def segment_sequence(sequence, window_size=35, step=5):
    """
    Teilt die Sequenz in Segmente (Fenster) der Länge window_size mit Schrittweite step.
    Gibt eine Liste von Tupeln (StartFrame, EndFrame, Segment) zurück.
    """
    segments = []
    num_frames = sequence.shape[0]
    for start in range(0, num_frames - window_size + 1, step):
        end = start + window_size
        segment = sequence[start:end]
        segments.append((start, end, segment))
    print(f"Erstellt {len(segments)} Segmente.")
    return segments

def compute_dtw_distance(seq1, seq2):
    """Berechnet die DTW-Distanz zwischen zwei Sequenzen."""
    distance, path = fastdtw(seq1, seq2, dist=euclidean)
    return distance, path


'''
def detect_punches_multiple(segments, ref_sequences):
    """
    Vergleicht jedes Segment mittels DTW mit allen Referenzsequenzen.
    Für jedes Segment wird der Schlagschlagtyp mit der geringsten DTW-Distanz ermittelt.
    Anschließend wird anhand des 10. Perzentils der besten Distanzen entschieden, 
    ob das Segment als Treffer gewertet wird.
    
    Rückgabe:
        - final_results: Liste von erkannten Schlägen als Tupel 
                         (StartFrame, EndFrame, Distance, Punch Type)
        - all_best_distances: Liste der minimalen DTW-Distanzen pro Segment.
    """
    
    thresholds = {"Hook body Left": 18, #sliding window problem: overlapping windows, no pause between punches 18
        "Hook head Left": 20, #sliding window + missing keypoints on left side 20
        "Hook head Right": 21, #21
        "Hook body Right": 21, #21
        "Front Kick Right": 25, #25
        "Front Kick Left": 37, #37
        "Straight Head Left": 20, #20
        "Straight Head Right": 18, #18
        "Low Kick Left": 24, #24
        "Low Kick Right": 28, #28
        "Roundhouse Kick Left": 30, #30
        "Roundhouse Kick Right": 30, #30
        "Side Kick Left": 45, #45
        "Side Kick Right": 40} #40
    results = []
    best_distances = []
    for start, end, segment in segments:
        best_punch = None
        best_distance = float('inf')
        for punch_type, ref_seq in ref_sequences.items():
            distance, _ = compute_dtw_distance(segment, ref_seq)
            """
            if distance < best_distance:
                best_distance = distance
                best_punch = punch_type
                """
            correct_threshold = thresholds[punch_type]
            if distance < correct_threshold:
                best_punch = punch_type
                best_distance = distance
                best_distances.append(best_distance)
                results.append((start, end, best_distance, best_punch))
    # Schwellwert anhand des 10. Perzentils der minimalen Distanzen bestimmen
    #threshold = 50.0
    #threshold = np.percentile(best_distances, 20)
    #final_results = [r for r in results if r[2] < threshold]
    final_results = results
    print("Erkannte Punches (Segmente):", final_results)
    return final_results, best_distances
    '''
    
def detect_punches_multiple(segments, ref_sequences):
    """
    Vergleicht jedes Segment mittels DTW mit allen Referenzsequenzen.
    Für jedes Segment wird der Schlagschlagtyp mit der geringsten DTW-Distanz ermittelt.
    Nur wenn diese Distanz unter dem schlagtypspezifischen Schwellenwert liegt, wird es als Treffer gewertet.
    """
    thresholds = {
        "Hook body Left": 18, #sliding window problem: overlapping windows, no pause between punches 18
        "Hook head Left": 20, #sliding window + missing keypoints on left side 20
        "Hook head Right": 21, #21
        "Hook body Right": 21, #21
        "Front Kick Right": 25, #25
        "Front Kick Left": 37, #37
        "Straight Head Left": 20, #20
        "Straight Head Right": 18, #18
        "Low Kick Left": 24, #24
        "Low Kick Right": 28, #28
        "Roundhouse Kick Left": 30, #30
        "Roundhouse Kick Right": 30, #30
        "Side Kick Left": 45, #45
        "Side Kick Right": 40  #40
        }

    results = []
    best_distances = []

    for start, end, segment in segments:
        min_distance = float('inf')
        best_punch = None

        # Berechne Distanz zu allen Referenzen und finde das Minimum
        for punch_type, ref_seq in ref_sequences.items():
            distance, _ = compute_dtw_distance(segment, ref_seq)
            
            if distance < min_distance:
                min_distance = distance
                best_punch = punch_type

        # Prüfe, ob die beste Distanz unter dem schlagtypspezifischen Schwellenwert liegt
        if best_punch and min_distance < thresholds.get(best_punch, float('inf')):
            results.append((start, end, min_distance, best_punch))
            best_distances.append(min_distance)

    print("Erkannte Punches (Segmente):", results)
    return results, best_distances

def save_weighted_results(results, filename="weighted_results.csv"):
    """Speichert die erkannten Ergebnisse in einer CSV-Datei."""
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Start Frame", "End Frame", "Distance", "Punch Type"])
        writer.writerows(results)
    print(f"Ergebnisse wurden in '{filename}' gespeichert.")

def annotate_video(video_path, output_path, results, flip_video=False):
    """
    Liest das Video erneut ein, ermittelt automatisch die Orientierung 
    und dreht es um 180°, wenn es kopfstehend ist.
    Schreibt den erkannten Schlagschlagtyp (sowie DTW-Distanz) auf die Frames,
    die innerhalb der erkannten Schlagintervalle liegen, und speichert 
    das annotierte Video als output_path.
    Wenn flip_video True ist, wird zusätzlich horizontal gespiegelt.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Fehler: Video konnte nicht geöffnet werden.")
        return

    # --- Automatische Orientierungserkennung ---
    rotate_video = False
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose_detector:
        found_frame = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_pose = pose_detector.process(image_rgb)
            if results_pose.pose_landmarks:
                landmarks = results_pose.pose_landmarks.landmark
                # Nase ist Landmark 0, Hüfte sind Landmark 23 und 24
                nose_y = landmarks[0].y
                hip1_y = landmarks[23].y
                hip2_y = landmarks[24].y
                hips_center_y = (hip1_y + hip2_y) / 2.0
                # In einem korrekt orientierten Bild sollte die Nase über dem Hüftzentrum liegen (kleinere y-Werte)
                if nose_y > hips_center_y:
                    rotate_video = True
                    print("Video scheint kopfstehend zu sein. Drehe das Ausgabevideo um 180°.")
                else:
                    print("Video-Orientierung scheint korrekt für das Ausgabevideo.")
                found_frame = True
                break
        # Setze den Video-Stream zurück an den Anfang
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # --- Ende der Orientierungserkennung ---

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Wende Rotation an, wenn das Video kopfstehend ist
        if rotate_video:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        
        if flip_video:
            frame = cv2.flip(frame, 1)
        
        # Prüfe, ob der aktuelle Frame in einem erkannten Schlagintervall liegt
        for start, end, dist, punch_type in results:
            if start <= frame_counter < end:
                # Zeichne ein Rechteck und die Textannotation
                top_left = (50, 50)
                bottom_right = (width - 50, height - 50)
                cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 3)
                elapsed_time = (frame_counter - start) / fps
                cv2.putText(frame, f"{punch_type}! ({dist:.2f})", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.putText(frame, f"Time: {elapsed_time:.1f}s", (50, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                # Add frame counter to the video
                cv2.putText(frame, f"Frame: {frame_counter}", (50, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                break
        
        out.write(frame)
        frame_counter += 1
    
    cap.release()
    out.release()
    print(f"Annotiertes Video wurde als '{output_path}' gespeichert.")

def plot_dtw_warping(seq1, seq2, path, title="DTW Warping Path"):
    """Visualisiert zwei Sequenzen und ihren Warping-Pfad mit Matplotlib."""
    plt.figure(figsize=(10, 6))
    
    # Sequenzen plotten
    plt.plot(seq1, label='Sequenz 1 (Referenz)', linestyle='--', alpha=0.7)
    plt.plot(seq2, label='Sequenz 2 (Test)', linestyle='--', alpha=0.7)
    
    # Warping-Pfad einzeichnen
    for (i, j) in path:
        plt.plot([i, j], [seq1[i], seq2[j]], color='gray', alpha=0.2)
    
    plt.title(title)
    plt.xlabel("Frame Index")
    plt.ylabel("Normalisierter Wert")
    plt.legend()
    plt.show()

def main():
    # 1. Video-Pfad abfragen
    video_file = get_video_file()
    if not video_file:
        print("Kein Video angegeben.")
        return
    print("Ausgewähltes Video:", video_file)
    
    # 2. Video verarbeiten: Pose-Daten extrahieren
    sequence = process_video(video_file)
    
    # 3. Mehrere Referenzsequenzen laden:
    # Hier definierst du ein Dictionary, in dem die Schlüssel die Schlagschlagtypen
    # und die Werte die Pfade zu den entsprechenden CSV-Dateien sind.
    """
    ref_csv_files = { #TODO: Pfade zu den Referenzsequenzen anpassen
        "Straight": "CSV:Schläge/DONE_straight_left_6.csv",
        "Low_Kick_Left": "CSV:Schläge/Ergebnisse Test Low Kick 9.csv",
        #"Left Hook": "/Pfad/zur/left_hook.csv",
        # Füge hier weitere Schlagschlagtypen und Pfade hinzu.

    }
    """
    ref_csv_files = {
        "Hook body Left": "DTW/references/hook_body_left.csv",
        "Hook head Left": "DTW/references/hook_left_head.csv",
        "Hook head Right": "DTW/references/hook_head_right.csv",
        "Hook body Right": "DTW/references/hook_body_right.csv",
        "Front Kick Right": "DTW/references/frontkick_right.csv",
        "Front Kick Left": "DTW/references/frontkick_left.csv",
        "Straight Head Left": "DTW/references/straight_left.csv",
        "Straight Head Right": "DTW/references/straight_right.csv",
        "Low Kick Left": "DTW/references/lowkick_left.csv",
        "Low Kick Right": "DTW/references/lowkick_right.csv",
        "Roundhouse Kick Left": "DTW/references/roundhouse_left.csv",
        "Roundhouse Kick Right": "DTW/references/roundhouse_right.csv",
        "Side Kick Left": "DTW/references/sidekick_left.csv",
        "Side Kick Right": "DTW/references/sidekick_right.csv",
    }
    ref_sequences = {}

    for punch_type, csv_file in ref_csv_files.items():
        print(f"Lade Referenzsequenz für '{punch_type}' aus '{csv_file}'")
        ref_sequences[punch_type] = load_reference_sequence(csv_file)
    
    # 4. Segmentierung der Video-Pose-Sequenz
    segments = segment_sequence(sequence, window_size=35, step=35)
    
    # 5. Erkennung von Schlägen via DTW mit mehreren Referenzen
    
    results, distances = detect_punches_multiple(segments, ref_sequences)

    # get unique punch types from results
    punch_type_counts = {}
    for start, end, dist, punch_type in results:
        if punch_type is not None:
            punch_type_counts[punch_type] = punch_type_counts.get(punch_type, 0) + 1
    
    print("Erkannte Schlagtypen und Anzahl der Erkennungen:")
    for punch_type, count in punch_type_counts.items():
        print(f"{punch_type}: {count}x")

    unique_punch_types = list(punch_type_counts.keys())
    print("Erkannte Schlagtypen:", unique_punch_types)

    # go through all results min and max frames and check if in the test csv file as the label is != 0
    # in this segment the count of the label which is not 0 should be greater than 15

    # test_csv = pd.read_csv('train/DONE_right_low_kick_7.csv')

    # count = 0
    # for start, end, dist, punch_type in results:
    #     if punch_type is not None and punch_type == "Low Kick Right":
    #         for i in range(start, end):
    #             if test_csv['label'][i] != 0:
    #                 count += 1
    #         if count > 35:
    #             print("Punch detected in the segment")
    #         else:
    #             print("Punch not detected in the segment")

    
    # 6. Ergebnisse in CSV speichern
    save_weighted_results(results)
    
    """
    # 7. Plot der DTW-Distanzen (Histogramm der minimalen Distanzen pro Segment)
    plt.hist(distances, bins=20)
    threshold = 50.0
    #threshold = np.percentile(distances, 20)
    plt.axvline(threshold, color='r', linestyle='dashed', linewidth=2, label=f'Threshold: {threshold:.2f}')
    plt.xlabel('DTW Distance')
    plt.ylabel('Frequency')
    plt.title('Distance Distribution')
    plt.legend()
    plt.show()
    """
    
    # 5.1 Beispielhafte Visualisierung für einen erkannten Schlag
    if len(results) > 0:
        # Nehme das erste erkannte Segment und die passende Referenz
        start, end, dist, punch_type = results[0]
        test_segment = sequence[start:end]
        ref_seq = ref_sequences[punch_type]
        
        # Wähle einen bestimmten Landmark (z.B. 27 für linke Hand)
        lm_idx = 15
        dim = 0  # x-Koordinate (0=x, 1=y, 2=z)
        
        # Extrahiere die Features für diesen Landmark
        seq1 = [frame[lm_idx*4 + dim] for frame in ref_seq]
        seq2 = [frame[lm_idx*4 + dim] for frame in test_segment]
        
        # Berechne DTW-Pfad
        distance, path = compute_dtw_distance(ref_seq, test_segment)
        
        '''
        # Einfache Visualisierung
        plot_dtw_warping(
            seq1, seq2, path, 
            title=f"Warping-Pfad für {punch_type} (Landmark {lm_idx}, X-Koordinate)"
        )
        
        
        # Erweiterte Visualisierung mit Subplots für X, Y, Z
        fig, axs = plt.subplots(3, 1, figsize=(12, 8))
        for dim, ax in zip([0, 1, 2], axs):
            seq1 = [frame[lm_idx*4 + dim] for frame in ref_seq]
            seq2 = [frame[lm_idx*4 + dim] for frame in test_segment]
            
            # Normalisierung der Sequenzen
            seq1 = (seq1 - np.mean(seq1)) / np.std(seq1)
            seq2 = (seq2 - np.mean(seq2)) / np.std(seq2)
            
            # Plotten auf dem spezifischen Subplot
            ax.plot(seq1, label='Referenz', linestyle='--', alpha=0.7)
            ax.plot(seq2, label='Test', linestyle='--', alpha=0.7)
            
            # Warping-Pfad einzeichnen
            for (i, j) in path:
                ax.plot([i, j], [seq1[i], seq2[j]], color='gray', alpha=0.2)
            
            ax.set_title(f"{punch_type} (Landmark {lm_idx}, {'X' if dim==0 else 'Y' if dim==1 else 'Z'}-Koordinate)")
            ax.set_xlabel("Frame Index")
            ax.set_ylabel("Normalisierter Wert")
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"dtw_plot_{punch_type}.png", dpi=300)
        plt.show()

        '''
    
    # 8. Annotiere das Video und speichere das Ergebnis
    output_video = "annotated_output.mp4"
    annotate_video(video_file, output_video, results)

if __name__ == "__main__":
    main()