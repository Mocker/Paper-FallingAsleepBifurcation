import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import h5py
import mne

def parse_nsrr_xml(xml_path, epoch_len=30):
    """
    Parses NSRR XML annotation file and returns epoch-by-epoch sleep stages
    to replicate loadPSGAnnotationClass and the alignment logic.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing XML {xml_path}: {e}")
        return None

    # NSRR XML structure typically has ScoredEvents/ScoredEvent
    stages = []
    durations = []
    starts = []
    
    arousals_start = []
    arousals_duration = []

    # Map EventConcept names to numbers if necessary
    # NSRR standard uses integers for stages (0=Wake, 1=N1, 2=N2, 3=N3, 4=N4, 5=REM)
    # or strings like "Stage 1 sleep|1"
    for event in root.findall('.//ScoredEvent'):
        concept = event.find('EventConcept').text
        start = float(event.find('Start').text)
        duration = float(event.find('Duration').text)

        if "stage" in concept.lower() or "sleep stage" in concept.lower() or "wake" in concept.lower() or "rem" in concept.lower():
            # Try to extract the integer representing the stage
            # E.g., "Stage 2 sleep|2" or just "2"
            try:
                stage_val = int(concept.split('|')[-1])
            except ValueError:
                # Fallback mapping
                concept_lower = concept.lower()
                if "wake" in concept_lower:
                    stage_val = 0
                elif "stage 1" in concept_lower or "n1" in concept_lower:
                    stage_val = 1
                elif "stage 2" in concept_lower or "n2" in concept_lower:
                    stage_val = 2
                elif "stage 3" in concept_lower or "n3" in concept_lower or "stage 4" in concept_lower:
                    stage_val = 3
                elif "rem" in concept_lower:
                    stage_val = 5
                else:
                    stage_val = np.nan
            
            stages.append(stage_val)
            starts.append(start)
            durations.append(duration)
        elif "arousal" in concept.lower():
            arousals_start.append(start)
            arousals_duration.append(duration)

    if not starts:
        return None

    starts = np.array(starts)
    durations = np.array(durations)
    stages = np.array(stages)

    # Reconstruct the continuous epoch scoring vector (replicates lines 263-282 of Data_load.m)
    # The scoring elements should be sequential, but there can be gaps
    score_len = len(starts)
    new_score_vec = []
    
    for i in range(score_len):
        num_epochs = int(durations[i] // epoch_len)
        if i < score_len - 1:
            # Check for gaps between current event and next event
            if starts[i] + durations[i] == starts[i+1]:
                new_score_vec.extend([stages[i]] * num_epochs)
            else:
                num_empty_epochs = int((starts[i+1] - (starts[i] + durations[i])) // epoch_len)
                new_score_vec.extend([stages[i]] * num_epochs)
                new_score_vec.extend([np.nan] * num_empty_epochs)
        else:
            new_score_vec.extend([stages[i]] * num_epochs)

    return {
        'scoringLabels': np.array(new_score_vec),
        'scoringStart': starts[0],
        'scoringPeriod': len(new_score_vec) * epoch_len,
        'scoreEpoch': epoch_len,
        'ArousalsStart': np.array(arousals_start),
        'ArousalsDuration': np.array(arousals_duration)
    }

def load_edf_data(edf_path, channels_to_load=['Fz-Cz', 'Cz-Oz', 'C4-M1', 'EKG'], hours_used=6):
    """
    Loads specific channels from EDF file and slices them to the required length.
    Replicates blockEdfLoad and indexing logic.
    """
    try:
        # exclude_channels is not strictly needed but we can pre-select to speed up loading
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        print(f"Error loading EDF {edf_path}: {e}")
        return None

    # Sampling rate check
    sfreq = raw.info['sfreq']
    
    # Calculate datapoints to take (e.g. 6 hours)
    time_to_take = hours_used * 60 * 60
    datapoints_used = int(time_to_take * sfreq)
    
    signals = {}
    channel_names_in_edf = raw.ch_names
    
    # Find matching channels (case-insensitive and flexible mapping)
    for target_ch in channels_to_load:
        matched_ch = None
        for ch in channel_names_in_edf:
            if target_ch.lower() in ch.lower().replace(" ", ""):
                matched_ch = ch
                break
        
        if matched_ch:
            # Slice signal
            sig_data = raw.get_data(picks=matched_ch).flatten()
            if len(sig_data) > datapoints_used:
                sig_data = sig_data[:datapoints_used]
            signals[target_ch] = sig_data
        else:
            print(f"Warning: Channel {target_ch} not found in {edf_path}")
            signals[target_ch] = None

    return signals, sfreq

def process_subject(subject_id, edf_dir, xml_dir, demo_df, output_h5_file):
    """
    Processes a single subject, extracts EDF signals and XML annotations,
    and writes them into a unified HDF5 file.
    """
    # Filename matching (mesa-sleep-XXXX.edf)
    prefix = 'mesa-sleep-'
    subject_str = f"{subject_id:04d}"
    edf_name = f"{prefix}{subject_str}.edf"
    xml_name = f"{prefix}{subject_str}-nsrr.xml"
    
    edf_path = os.path.join(edf_dir, edf_name)
    xml_path = os.path.join(xml_dir, xml_name)
    
    if not os.path.exists(edf_path):
        print(f"EDF file not found: {edf_path}")
        return False
        
    print(f"Processing Subject {subject_id}...")
    
    # 1. Load EDF Signals
    signals_data = load_edf_data(edf_path)
    if not signals_data:
        return False
    signals, sfreq = signals_data
    
    # 2. Parse XML Annotations
    annotations = None
    if os.path.exists(xml_path):
        annotations = parse_nsrr_xml(xml_path)
    else:
        print(f"Warning: XML annotation not found for subject {subject_id}")
        
    # 3. Save to HDF5
    with h5py.File(output_h5_file, 'a') as f:
        # Create group for this subject
        grp_name = f"subject_{subject_str}"
        if grp_name in f:
            del f[grp_name] # Overwrite
            
        grp = f.create_group(grp_name)
        grp.attrs['original_id'] = subject_id
        
        # Save signals
        sig_grp = grp.create_group('signals')
        sig_grp.attrs['sampling_rate'] = sfreq
        for ch_name, data in signals.items():
            if data is not None:
                sig_grp.create_dataset(ch_name, data=data, compression='gzip')
                
        # Save annotations
        if annotations:
            annot_grp = grp.create_group('annotations')
            for key, val in annotations.items():
                if val is not None:
                    annot_grp.create_dataset(key, data=val)
                    
        # Save demographic metadata
        sub_demo = demo_df[demo_df['MESA_EDFid'] == subject_id]
        if not sub_demo.empty:
            demo_grp = grp.create_group('demographics')
            demo_grp.attrs['age'] = sub_demo['Age'].values[0]
            demo_grp.attrs['gender'] = sub_demo['Gender'].values[0]
            demo_grp.attrs['race'] = sub_demo['Race'].values[0]
            
    print(f"Subject {subject_id} processed successfully.")
    return True

if __name__ == "__main__":
    # Example usage:
    # process_subject(1, 'path/to/edfs', 'path/to/xmls', demo_df, 'mesa_dataset.h5')
    pass
