#!/usr/bin/env python
# -+*- coding: utf-8 -*-
import json
import os
import mariadb


def insert_log_data_into_mariadb(log_file_path):
    """
    Reads a JSONL log file and inserts each record into a MariaDB table.
    """
    # 1. MariaDB Connection Details
    db_config = {
        'host': '',
        'user': '',
        'password': '',
        'database': ''
    }

    try:
        # 2. Establish database connection
        print("Connecting to the MariaDB database...")
        conn = mariadb.connect(**db_config)
        cursor = conn.cursor()
        print("Connection successful!")

        # 3. SQL INSERT statement
        insert_query = """
        INSERT INTO image_performance_log (
            image_file, 
            query, 
            response, 
            generated_tokens, 
            generation_time_seconds, 
            tokens_per_second
        ) VALUES (%s, %s, %s, %s, %s, %s)
        """

        # 4. Read the log file and insert records
        records_inserted = 0
        print(f"Reading log file: {log_file_path}")
        with open(log_file_path, 'r') as log_file:
            # Skip the initial header lines
            for line in log_file:
                if line.strip().startswith('---'):
                    continue
                if line.strip().startswith('Timestamp:'):
                    continue
                if not line.strip():  # Skip empty lines
                    continue

                try:
                    # Parse the JSON record
                    record = json.loads(line.strip())

                    # Prepare the data for insertion
                    data = (
                        record.get("image_file"),
                        record.get("query"),
                        record.get("response"),
                        record.get("generated_tokens"),
                        record.get("generation_time_seconds"),
                        record.get("tokens_per_second", None)  # Handle potential missing key
                    )

                    # Execute the insert statement
                    cursor.execute(insert_query, data)
                    records_inserted += 1

                except json.JSONDecodeError as e:
                    print(f"Skipping malformed JSON line: {line.strip()}. Error: {e}")

        # 5. Commit the transaction
        conn.commit()
        print(f"\nSuccessfully inserted {records_inserted} records into the database.")

    except mariadb.Error as err:
        if err.errno == mariadb.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password.")
        elif err.errno == mariadb.ER_BAD_DB_ERROR:
            print("Database does not exist.")
        else:
            print(f"Database error: {err}")
    except FileNotFoundError:
        print(f"Error: The file '{log_file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    finally:
        # 6. Close the connection
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()
            print("Database connection closed.")


if __name__ == "__main__":
    # Define the path to your log file
    log_file_name = ("../conversation/batch_performance_log")
    script_dir = os.path.dirname(__file__)
    log_file_path = os.path.join(script_dir, log_file_name)

    insert_log_data_into_mariadb(log_file_path)