import mariadb
import configparser
import os
from mariadb.constants import ERR



# 1. Load database configuration from the db_config.ini file
def load_db_config(config_file_path, table):
    config = configparser.ConfigParser()
    if not os.path.exists(config_file_path):
        print(f"Error: Configuration file '{config_file_path}' not found.")
        return None
    config.read(config_file_path)
    if 'Database' in config:
        table = config['Database'].get('db_table', 'performance_log_vlm')  # Default table name
        return {
            'host': config['Database'].get('db_host'),
            'user': config['Database'].get('db_user'),
            'password': config['Database'].get('db_password'),
            'database': config['Database'].get('db_name', 'ha'),
        }
    else:
        print("Error: No '[Database]' section found in the config file.")
        return None

# Get the path to db_config.ini relative to the current script
script_dir = os.path.dirname(__file__)
config_path = os.path.join(script_dir, "../conversation/db_config.ini")
table = "performance_log_vlm"  # Default table name if not specified in the config
db_config = load_db_config(config_path, table)

if db_config:
    conn = None
    try:
        # 2. Establish database connection
        print("Attempting to connect to the MariaDB database...")
        conn = mariadb.connect(**db_config)
        cursor = conn.cursor()
        print("Connection successful!")

        # 3. Your database operations go here
        # Example: a query that might fail if the table doesn't exist
        query = f"SELECT * FROM {table}"
        cursor.execute(query)

    except mariadb.Error as err:
        # 4. Use the ERR constants to handle specific errors
        if err.errno == ERR.ER_ACCESS_DENIED_ERROR:
            print(f"Connection Error: Access denied for user '{db_config['user']}'. Please check the username and password.")
        elif err.errno == ERR.ER_BAD_DB_ERROR:
            print(f"Connection Error: The database '{db_config['database']}' does not exist.")
        elif err.errno == ERR.ER_NO_SUCH_TABLE:
            print(f"Query Error: The table {table} does not exist. Error code: {err.errno}")
        else:
            # Handle all other MariaDB errors
            print(f"An unexpected MariaDB error occurred: {err}")

    except FileNotFoundError:
        print("Error: The 'db_config.ini' file was not found.")

    except Exception as e:
        # Catch any other unexpected Python errors
        print(f"An unexpected error occurred: {e}")

    finally:
        # 5. Ensure the connection is always closed
        # The 'is_connected()' check is not available on the mariadb.Connection object
        if conn:
            conn.close()
            print("Database connection closed.")