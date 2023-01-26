import os
import pickle

from sympy import simplify_logic


def get_explanations(i, labels):
    simple_concept_dict = {}
    print("================")
    print(i)
    print(labels[i])
    print("================")
    explanations = []
    simple_logic = None
    for explanation_raw in concept_dict[i]:
        explanations.append(explanation_raw)

        # aggregate example-level explanations

        aggregated_explanation = " | ".join(explanations)
        if i == 6:
            simple_logic = simplify_logic(aggregated_explanation, "dnf")
        else:
            simple_logic = simplify_logic(aggregated_explanation, "dnf", force=True)

    print(simple_logic)
    simple_concept_dict[labels[i]] = simple_logic

    pickle.dump(
        simple_concept_dict,
        open(
            os.path.join(
                output_path_per_class, f"explanation_bird_class_horse.pkl"
            ),
            "wb",
        ),
    )


if __name__ == '__main__':
    labels = [
        "antelope", "grizzly+bear", "killer+whale", "beaver", "dalmatian", "persian+cat", "horse",
        "german+shepherd", "blue+whale", "siamese+cat", "skunk", "mole", "tiger", "hippopotamus",
        "leopard", "moose", "spider+monkey", "humpback+whale", "elephant", "gorilla", "ox",
        "fox", "sheep", "seal", "chimpanzee", "hamster", "squirrel", "rhinoceros", "rabbit", "bat",
        "giraffe", "wolf", "chihuahua", "rat", "weasel", "otter", "buffalo", "zebra", "giant+panda",
        "deer", "bobcat", "pig", "lion", "mouse", "polar+bear", "collie",
        "walrus", "raccoon", "cow", "dolphin"
    ]
    output_path = "root-path/out/awa2/Baseline_PostHoc/ViT-B_16/lr_0.01_epochs_500_temperature-lens_6.0_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.45_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_10.0_hidden-layers_1_layer_VIT_explainer_init_none"
    path = os.path.join(
        output_path,
        "cub-baseline_explanations.pkl",
    )
    print(path)
    results_arr = pickle.load(open(path, "rb"))
    concept_label = []
    num_concepts_ex = []
    concept_dict = {}
    y_unique = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
    print(y_unique)
    for y in y_unique:
        concept_dict[int(y)] = []
    for results in results_arr:
        concept_label.append(results["dict_sample_concept"])
        num_concepts_ex.append(results["num_concepts"])
        if results["concept_dict_key"] != -1:
            concept_dict[results["concept_dict_key"]].append(results["concept_dict_val"])
    best_explanation = ""
    keys = list(concept_dict.keys())
    # Get Explanation by label id
    output_path_per_class = os.path.join(output_path, "Explanations_per_class")
    os.makedirs(output_path_per_class, exist_ok=True)

    get_explanations(i=labels.index("otter"), labels=labels)
    get_explanations(i=labels.index("antelope"), labels=labels)
    get_explanations(i=labels.index("beaver"), labels=labels)
    # get_explanations(i=labels.index("moose"), labels=labels)
