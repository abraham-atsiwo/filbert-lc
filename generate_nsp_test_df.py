from pipeline import generate_test_from_bloombery


# Protect the entry point of the program
if __name__ == "__main__":
    save_path = "data/nsp/bloombery_test_long_nsp.csv"
    root_directory = "data/nsp/bloombery_test_df"
    generate_test_from_bloombery(root_directory=root_directory, save_path=save_path)
