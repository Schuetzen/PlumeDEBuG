import os
import sys
import json
import sqlite3
import numpy as np
import scipy.io
import h5py

def load_mat_file(matlab_file):
    """
    Load MATLAB file, supporting both v7 (and earlier) and v7.3 formats.
    
    Parameters:
        matlab_file: MATLAB file path
    
    Returns:
        tuple: (mat_data, is_v73)
            - mat_data: Dictionary if v7 format; None if v7.3 format
            - is_v73: Boolean indicating if it's v7.3 format
    """
    try:
        # Try to read v7 and earlier versions using scipy.io.loadmat
        mat_data = scipy.io.loadmat(matlab_file, squeeze_me=True, struct_as_record=False)
        return mat_data, False
    except NotImplementedError:
        print("Detected v7.3 MATLAB file, using h5py for reading")
        return None, True
    except Exception as e:
        raise Exception(f"Failed to load MATLAB file: {e}")

def safe_extract(data, f):
    """
    Recursively parse h5py.Reference objects.
    
    Parameters:
        data: Data to be parsed
        f: Currently open h5py.File object, used for dereferencing
    
    Returns:
        Parsed data that can be serialized to JSON
    """
    if isinstance(data, h5py.Reference):
        if data:
            obj = f[data]
            return safe_extract(obj[()], f)
        else:
            return None
    elif isinstance(data, np.ndarray):
        # If it's an object array, parse each item recursively
        if data.dtype.kind == 'O':
            return [safe_extract(item, f) for item in data]
        else:
            # Otherwise convert directly to Python list
            return data.tolist()
    elif isinstance(data, (list, tuple)):
        return [safe_extract(item, f) for item in data]
    else:
        return data

def maybe_decode_ascii(data):
    """
    If data is a list of ASCII codes (or list of lists), attempt to convert to string;
    otherwise return as is.
    """
    # If it's a 2D list, e.g., [[73], [109], [103], ...]
    if isinstance(data, list) and len(data) > 0:
        if all(isinstance(x, list) and len(x) == 1 and isinstance(x[0], (int, np.integer)) for x in data):
            data = [x[0] for x in data]  # Flatten
        # If it's a list of numbers (considering np.integer types), try to convert to string
        if all(isinstance(x, (int, np.integer)) for x in data):
            try:
                return ''.join(chr(int(x)) for x in data)
            except ValueError:
                pass
    return data

def fix_field_value(field, data):
    """
    Convert specific fields:
      - imageName and folderPath: Decode ASCII code list to string if needed; otherwise join list elements
      - bubble_count converted to int
      - bubble_diameter converted to float
      - Other fields remain unchanged
    """
    data = maybe_decode_ascii(data)
    
    if field in ['imageName', 'folderPath']:
        if isinstance(data, list):
            if len(data) == 1:
                data = data[0]
            else:
                data = ''.join(str(x) for x in data)
        return str(data)
    elif field == 'bubble_count':
        try:
            return int(data)
        except:
            return 0
    elif field == 'bubble_diameter':
        try:
            return float(data)
        except:
            return 0.0
    else:
        return data


def extract_imginfo_from_v7(mat_data):
    """
    Extract fields from imgInfo structure in a v7 MATLAB file dictionary
    
    Parameters:
        mat_data: Dictionary obtained from scipy.io.loadmat
    
    Returns:
        list: Each record is a dictionary containing imageName, bubble_count, bubble_diameter,
              bubble_positions, bounding_boxes, folderPath, etc.
    """
    if 'imgInfo' not in mat_data:
        raise ValueError("'imgInfo' variable does not exist in the MATLAB file")
    
    img_info = mat_data['imgInfo']
    # If there's only a single record, wrap it in an array for uniform processing
    if not isinstance(img_info, np.ndarray):
        img_info = np.array([img_info])
    
    records = []
    for rec in img_info:
        record = {}
        record['imageName'] = fix_field_value('imageName', getattr(rec, 'imageName', ''))
        record['bubble_count'] = fix_field_value('bubble_count', getattr(rec, 'bubble_count', 0))
        record['bubble_diameter'] = fix_field_value('bubble_diameter', getattr(rec, 'bubble_diameter', 0.0))

        bp = getattr(rec, 'bubble_positions', [])
        if isinstance(bp, np.ndarray):
            bp = bp.tolist()
        record['bubble_positions'] = bp

        bb = getattr(rec, 'bounding_boxes', None)
        if bb is None:
            bb = getattr(rec, 'bouding_boxes', [])
        if isinstance(bb, np.ndarray):
            bb = bb.tolist()
        record['bounding_boxes'] = bb

        record['folderPath'] = fix_field_value('folderPath', getattr(rec, 'folderPath', ''))

        records.append(record)
    return records

def extract_imginfo_from_v73(matlab_file):
    """
    Use h5py to extract fields from imgInfo in a v7.3 MATLAB file.
    If multiple records exist, extract each one.
    
    Parameters:
        matlab_file: MATLAB file path
    
    Returns:
        list: Each record is a dictionary
    """
    records = []
    try:
        with h5py.File(matlab_file, 'r') as f:
            if 'imgInfo' not in f:
                raise ValueError("'imgInfo' group does not exist in the MATLAB file")
            imgInfo_group = f['imgInfo']

            # Define fields to extract and their default values
            fields = [
                ('imageName', ''),
                ('bubble_count', 0),
                ('bubble_diameter', 0.0),
                ('bubble_positions', []),
                ('bounding_boxes', []),
                ('folderPath', '')
            ]
            
            # Determine if there are multiple records (assuming all fields have the same record count)
            record_count = 1
            sample_field = fields[0][0]  # e.g., "imageName"
            if sample_field in imgInfo_group:
                data_sample = safe_extract(imgInfo_group[sample_field][()], f)
                # If data_sample is a list and length > 1, there are multiple records
                if isinstance(data_sample, list) and len(data_sample) > 1:
                    record_count = len(data_sample)
            
            for i in range(record_count):
                record = {}
                for field, default in fields:
                    if field in imgInfo_group:
                        data = safe_extract(imgInfo_group[field][()], f)
                        # If it's a list and there are multiple records, take the i-th one
                        if isinstance(data, list) and record_count > 1:
                            try:
                                data = data[i]
                            except IndexError:
                                data = default
                        data = fix_field_value(field, data)
                        record[field] = data
                    else:
                        record[field] = default
                records.append(record)

    except Exception as e:
        print(f"Error processing v7.3 file: {e}")
    return records

def convert_to_sqlite(matlab_file, sqlite_db):
    """
    Extract fields from imgInfo in a .mat file and store them in a SQLite database
    
    Parameters:
        matlab_file: .mat file path (containing only an imgInfo variable)
        sqlite_db: Path to the SQLite database to create or update
    """
    try:
        print(f"Loading MATLAB file: {matlab_file}")
        mat_data, is_v73 = load_mat_file(matlab_file)
        if is_v73:
            records = extract_imginfo_from_v73(matlab_file)
        else:
            records = extract_imginfo_from_v7(mat_data)
        
        if not records:
            print("No records extracted")
            return False
        
        print(f"Extracted {len(records)} records")
        
        conn = sqlite3.connect(sqlite_db)
        cursor = conn.cursor()
        # Create or reuse a table with 7 columns
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_name TEXT,
                bubble_count INTEGER,
                bubble_diameter REAL,
                bubble_positions TEXT,
                bounding_boxes TEXT,
                folder_path TEXT
            )
        ''')
        
        for i, rec in enumerate(records):
            try:
                image_name = rec.get('imageName', '')
                bubble_count = rec.get('bubble_count', 0)
                bubble_diameter = rec.get('bubble_diameter', 0.0)

                bp = rec.get('bubble_positions', [])
                bp_json = json.dumps(bp, ensure_ascii=False)  # Convert to JSON

                bb = rec.get('bounding_boxes', [])
                bb_json = json.dumps(bb, ensure_ascii=False)  # Convert to JSON

                folder_path = rec.get('folderPath', '')

                cursor.execute('''
                    INSERT INTO images (
                        image_name, 
                        bubble_count,
                        bubble_diameter, 
                        bubble_positions, 
                        bounding_boxes, 
                        folder_path
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (image_name, bubble_count, bubble_diameter, bp_json, bb_json, folder_path))
            except Exception as e:
                print(f"Error processing record {i}: {e}")
        
        conn.commit()
        conn.close()
        print(f"Conversion complete, data saved to {sqlite_db}")
        return True
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False

def query_example(sqlite_db):
    """
    Example query: Display some records stored in the SQLite database
    """
    if not os.path.exists(sqlite_db):
        print(f"Database file {sqlite_db} does not exist.")
        return
    conn = sqlite3.connect(sqlite_db)
    cursor = conn.cursor()
    print("Querying first 5 records:")
    cursor.execute("SELECT image_name, bubble_count, bubble_diameter FROM images LIMIT 5")
    for row in cursor.fetchall():
        print(f"Image: {row[0]}, Bubble count: {row[1]}, Bubble diameter: {row[2]}")
    conn.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isdir(path):
            processed = 0
            errors = 0
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.mat') and 'aggregated_results' in file:
                        matlab_file = os.path.join(root, file)
                        sqlite_db = os.path.join(root, "bubbleDB.db")
                        print(f"\nProcessing file: {matlab_file}")
                        if convert_to_sqlite(matlab_file, sqlite_db):
                            processed += 1
                        else:
                            errors += 1
            print(f"\nSuccessfully processed {processed} files, {errors} files had errors")
        elif os.path.isfile(path) and path.endswith('.mat'):
            matlab_file = path
            sqlite_db = os.path.splitext(matlab_file)[0] + ".db"
            if convert_to_sqlite(matlab_file, sqlite_db):
                query_example(sqlite_db)
        else:
            print(f"Error: {path} is not a valid directory or .mat file")
            sys.exit(1)
    else:
        print("Please specify a .mat file or directory path, for example:")
        print(f"  python {sys.argv[0]} path/to/aggregated_results.mat")
        print(f"  python {sys.argv[0]} path/to/directory")