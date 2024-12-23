from datetime import datetime
import os   

# function to find the latest file in the directory
def churn_filename():
    # Construct the directory path
    directory = os.path.join(os.getcwd() , "Telco Customer Data")
    
    # Initialize the latest date and file
    latest_file_date = datetime.strptime("01011900", "%m%d%Y")  # Start with an old date
    latest_file = None

    try:
        # Iterate through files in the directory
        for entry in os.scandir(directory):
            if entry.is_file() and entry.name.startswith("Telco_Customer_DataSet_"):
                # Extract the date portion from the filename (assumes format like "Telco_Customer_DataSet_mmddyyyy.csv")
                try:
                    file_date_str = entry.name[-12:-4]  # Extract "ddmmyyyy" from the filename
                    file_date = datetime.strptime(file_date_str, "%m%d%Y")
                except ValueError:
                    # Skip files with invalid date formats
                    continue
                
                # Compare with the latest file date
                if file_date > latest_file_date:
                    latest_file_date = file_date
                    latest_file = entry.name
    except FileNotFoundError:
        print(f"Directory not found: {directory}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return latest_file