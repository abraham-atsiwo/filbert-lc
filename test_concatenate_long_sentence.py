from pipeline import parse_args, generate_long_sentence


if __name__ == "__main__":
    # data = pd.read_csv("sentences_50agree/train.csv")
    args = parse_args()
    print(args)

    if not args.is_terminal_args:
        print("Not Terminal Args")
    for c_size in range(8, 11):
        for lbl in range(1,2):
            generate_long_sentence(
                data=None,
                device="cpu",
                concatenate=True,
                save_filename=f"sentences_50agree_test_{c_size}.csv",
                label=lbl,
                num_comb=c_size,
                data_path="data/sentences_50agree/test.csv",
                model_name="../saved_models/lambda-labs-nsp-200000",
               concatenate_path='data/concat_data_test_history',
                n_core=8,
            )
    else:
        print("Terminal Args")
        generate_long_sentence(
            data_path=args.data_path,
            device=args.device,
            concatenate=args.concatenate,
            save_filename=args.save_filename,
            label=args.label,
            num_comb=args.num_comb,
            n_core=args.n_core,
            model_name=args.model_name,
            concatenate_path='concat_data_test_history'
        )
