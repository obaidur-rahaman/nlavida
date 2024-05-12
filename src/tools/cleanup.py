
import os

def cleanUserDescription(root_directory):
    prompt_directory = root_directory + "prompt/"
    # Now delete the user description file
    # print(prompt_directory + "user_description_of_file.txt")
    if os.path.exists(prompt_directory + "user_description_of_file.txt"):
        os.remove(prompt_directory + "user_description_of_file.txt")
        print("User description file deleted successfully")
