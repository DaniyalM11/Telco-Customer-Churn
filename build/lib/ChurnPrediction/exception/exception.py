import sys
from ChurnPrediction.custom_logging import logger

class ChurnPredictionException(Exception):
    def __init__(self,error_message, error_details:sys):
        self.error_message = error_message
        _,_,exc_tb = error_details.exc_info()
        
        self.lineno = exc_tb.tb_lineno
        self.filename = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return f"Error: {str(self.error_message)} at {self.filename} line {self.lineno}"
    
if __name__ == "__main__":
    try:
        logger.logging.info("Enter the try block")
        a=1/0
        print("This will not be printed",a)
    except Exception as e:
        raise ChurnPredictionException(e,sys)        

