import shutil
import os
import trajectory_tools.templates
from importlib_resources import files


def main():
    print("Initializing trajectory workspace")
    template_path = files(trajectory_tools.templates)
    for file in template_path.iterdir():
        print(f"Copying {str(file)} to {os.getcwd()}...")
        shutil.copy(str(file), os.getcwd())
    print("Initialization complete. Execute `trajectory_edit` or `trajectory_sim`.")


if __name__ == "__main__":
    main()
