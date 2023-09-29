from typing import Union

def minority_strategy(occurences: dict):
    min_occurences = min(occurences.values())
    max_occurences = max(occurences.values())
    occurences_gap = max_occurences - min_occurences
    new_occurences = {
        class_: occurences_gap
                if occurences[class_] == min_occurences else 0
                for class_ in occurences.keys()
    }
    return new_occurences


def not_minority_strategy(occurences: dict):
    min_occurences = min(occurences.values())
    max_occurences = max(occurences.values())
    new_occurences = {
        class_: max_occurences - occurences[class_]
                if occurences[class_] != min_occurences else 0
                for class_ in occurences.keys()
    }
    return new_occurences


def not_majority_strategy(occurences: dict):
    max_occurences = max(occurences.values())
    new_occurences = {
        class_: max_occurences - occurences[class_]
                for class_ in occurences.keys()
    }
    return new_occurences


def all_strategy(occurences: dict):
    max_occurences = max(occurences.values())
    new_occurences = {
        class_: max_occurences - occurences[class_]
                for class_ in occurences.keys()
    }
    return new_occurences

def threshold_strategy(occurences: dict, threshold: int):
    max_occurences = max(occurences.values())
    new_occurences = {
        class_: min(max_occurences - occurences[class_], occurences[class_]*threshold)
                for class_ in occurences.keys()
    }
    return new_occurences


###############################################################################

def oversampling_strategy(
    occurences: dict, strategy: Union[str, int, float] = "auto"
):
    n_occurences = sum([n for n in occurences.values()])
    perfectly_balanced_occurences = int(n_occurences / len(occurences.keys()))

    if strategy == "auto":
        n_generate = {
            class_: perfectly_balanced_occurences - occ
                    if occ < perfectly_balanced_occurences else 0
                    for class_, occ in occurences.items()
        }

    else:
        n_generate = {
            class_: int(min(occ * strategy, perfectly_balanced_occurences - occ))
            if occ < perfectly_balanced_occurences else 0
            for class_, occ in occurences.items()
        }

    return n_generate

def undersampling_strategy(
    occurences: dict, strategy: Union[str, int, float] = "auto"
):
    n_occurences = sum([n for n in occurences.values()])
    perfectly_balanced_occurences = int(n_occurences / len(occurences.keys()))

    if strategy == "auto":
        n_remove = {
            class_: occ - perfectly_balanced_occurences
                    if occ > perfectly_balanced_occurences else 0
                    for class_, occ in occurences.items()
        }

    else:
        n_remove = {
            class_: int(min(occ * strategy, occ - perfectly_balanced_occurences))
            if occ > perfectly_balanced_occurences else 0
            for class_, occ in occurences.items()
        }

    return n_remove



if __name__ == "__main__":
    occ = {
        0: 806,
        1: 403,
        2: 206,
        3: 104,
        4: 51,
        5: 29
    }

    oversampling_thresholds = [0, 0.25, 0.5, 1, 5, "auto"]
    undersampling_thresholds = [0, 0.05, 0.1, 0.2, 0.3, "auto"]

    for over_t in oversampling_thresholds:
        for under_t in undersampling_thresholds:
            n_drop = undersampling_strategy(occ, under_t)
            n_gen = oversampling_strategy(occ, over_t)

            total = {k: int(occ[k] - n_drop[k] + n_gen[k]) for k in occ.keys()}

            print(f"OVER: {over_t} == UNDER: {under_t}")
            print(f"Original: \t{occ}")
            print(f"DROP: \t\t{n_drop}")
            print(f"GENERATE: \t{n_gen}")
            print(f"TOTAL: \t\t{total}")
            print("\n=========================\n")