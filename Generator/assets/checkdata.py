import sqlite3

def check_first_five_records(db_path):
    try:
        # Connect to SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # First check what tables exist in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables in database: {[table[0] for table in tables]}")
        
        # Query first 5 records from the images table (not bubbles)
        query = "SELECT id, image_name, bubble_count, bubble_diameter FROM images LIMIT 5;"
        cursor.execute(query)
        rows = cursor.fetchall()
        
        # Output query results
        print("\nFirst 5 records from images table:")
        for row in rows:
            print(f"ID: {row[0]}, Image: {row[1]}, Bubble count: {row[2]}, Diameter: {row[3]}")
        
        # Optionally check a single full record with all columns
        if rows:
            cursor.execute("SELECT * FROM images WHERE id = ?", (rows[0][0],))
            full_row = cursor.fetchone()
            print("\nFull details of first record:")
            print(f"ID: {full_row[0]}")
            print(f"Image name: {full_row[1]}")
            print(f"Bubble count: {full_row[2]}")
            print(f"Bubble diameter: {full_row[3]}")
            print(f"Bubble positions: {full_row[4][:100]}... (truncated)")
            print(f"Bounding boxes: {full_row[5][:100]}... (truncated)")
            print(f"Folder path: {full_row[6]}")
            
    except sqlite3.Error as e:
        print("SQLite error:", e)
    finally:
        # Close database connection
        if conn:
            conn.close()

if __name__ == "__main__":
    db_path = "../../Results/TestDB/aggregated_results.db"
    check_first_five_records(db_path)