"""file that creates a custom exception class and will be later used for logging"""
import sys
from logger import logging

def error_message_detail(error, error_detail: sys):
    """returns the error message encapsulated in Exception """
    # exc_info will tell you on which file,line the exception has occured
    _, _, exc_tb = error_detail.exc_info()
    # Check custom exception handling documentation for more info
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script name [{0}] and line number [{1}], error message is [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    return error_message


class CustomException(Exception):
    """ code block for custom exception"""
    def __init__(self, error_message, error_detail: sys):
        """ initializer """
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self):
        """returns the error message"""
        return self.error_message

