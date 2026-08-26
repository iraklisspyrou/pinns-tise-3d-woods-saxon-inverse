"""Dataset loading and fixed-cardinality state selection."""

import numpy as np

def load_fd_dataset(path="data/wahlborn_synthetic_dataset.npz"):
    raw = np.load(path, allow_pickle=True)["data"]
    return list(raw)


def get_sample_by_nucleus(dataset, A, Z, is_proton, max_states=6):
    """
    Pair-preserving state selector.

    Selection priority:
      1. deepest bound state,
      2. spin-orbit pairs grouped by (nr,l),
      3. unpaired states,
      4. remaining unused states.

    No fake states are created.
    """
    for item in dataset:
        if (
            int(item["A"]) == int(A)
            and int(item["Z"]) == int(Z)
            and bool(item["is_proton"]) == bool(is_proton)
        ):
            all_states = [
                {
                    "nr": int(st["nr"]),
                    "l": int(st["l"]),
                    "j": float(st["j"]),
                    "energy": float(st["energy"]),
                }
                for st in item["states"]
            ]

            if len(all_states) == 0:
                raise ValueError(
                    f"No states found for A={A}, Z={Z}, is_proton={is_proton}"
                )

            deepest_state = sorted(all_states, key=lambda x: float(x["energy"]))[0]
            selected_states = [deepest_state]

            def state_id(st):
                return (
                    int(st["nr"]),
                    int(st["l"]),
                    float(st["j"]),
                    float(st["energy"]),
                )

            selected_ids = {state_id(deepest_state)}

            groups = {}
            for st in all_states:
                key = (int(st["nr"]), int(st["l"]))
                groups.setdefault(key, []).append(st)

            pair_groups = [grp for grp in groups.values() if len(grp) == 2]
            unpaired_states = [
                st for grp in groups.values() if len(grp) == 1 for st in grp
            ]

            sorted_pairs = sorted(
                pair_groups,
                key=lambda grp: sum(float(s["energy"]) for s in grp) / 2.0,
            )

            remaining_slots = max_states - len(selected_states)
            max_pairs = remaining_slots // 2

            if max_pairs > 0:
                if len(sorted_pairs) > max_pairs:
                    target_deep_pairs = (max_pairs * 2) // 3
                    if target_deep_pairs == 0 and max_pairs > 0:
                        target_deep_pairs = 1

                    shallow_pairs_count = max_pairs - target_deep_pairs
                    deep_pairs = sorted_pairs[:target_deep_pairs]
                    shallow_pairs = (
                        sorted_pairs[-shallow_pairs_count:]
                        if shallow_pairs_count > 0
                        else []
                    )
                    chosen_pairs = deep_pairs + shallow_pairs
                else:
                    chosen_pairs = sorted_pairs

                for grp in chosen_pairs:
                    for st in grp:
                        if len(selected_states) >= max_states:
                            break
                        if state_id(st) not in selected_ids:
                            selected_states.append(st)
                            selected_ids.add(state_id(st))

            needed_states = max_states - len(selected_states)

            if needed_states > 0 and unpaired_states:
                unpaired_states = sorted(
                    unpaired_states,
                    key=lambda x: float(x["energy"]),
                )
                for st in unpaired_states:
                    if len(selected_states) >= max_states:
                        break
                    if state_id(st) not in selected_ids:
                        selected_states.append(st)
                        selected_ids.add(state_id(st))

            if len(selected_states) < max_states:
                remaining_states = sorted(
                    all_states,
                    key=lambda x: float(x["energy"]),
                )
                for st in remaining_states:
                    if len(selected_states) >= max_states:
                        break
                    if state_id(st) not in selected_ids:
                        selected_states.append(st)
                        selected_ids.add(state_id(st))

            selected_states = sorted(
                selected_states,
                key=lambda x: float(x["energy"]),
            )

            species_name = "proton" if is_proton else "neutron"
            print(f"\nSelected states for A={A}, Z={Z}, species={species_name}:")
            for idx, state in enumerate(selected_states, start=1):
                print(
                    f"  {idx:2d}. "
                    f"nr={int(state['nr'])}, "
                    f"l={int(state['l'])}, "
                    f"j={float(state['j']):.1f}, "
                    f"E={float(state['energy']):.6f} MeV"
                )

            return {
                "A": int(A),
                "Z": int(Z),
                "is_proton": bool(is_proton),
                "states": selected_states,
            }

    raise ValueError(f"No sample found for A={A}, Z={Z}, is_proton={is_proton}")


def list_available_cases(dataset_path="data/wahlborn_synthetic_dataset.npz"):
    dataset = load_fd_dataset(dataset_path)
    return [(int(x["A"]), int(x["Z"]), bool(x["is_proton"])) for x in dataset]


def build_multinucleus_samples(
    dataset_path="data/wahlborn_synthetic_dataset.npz",
    cases=None,
    max_states=6,
    require_exact_states=True,
):
    dataset = load_fd_dataset(dataset_path)
    if cases is None:
        cases = list_available_cases(dataset_path)

    samples, skipped = [], []
    for A, Z, is_proton in cases:
        try:
            sample = get_sample_by_nucleus(
                dataset,
                A=A,
                Z=Z,
                is_proton=is_proton,
                max_states=max_states,
            )
            if require_exact_states and len(sample["states"]) != max_states:
                skipped.append((A, Z, is_proton, len(sample["states"])))
                continue
            samples.append(sample)
        except Exception as e:
            skipped.append((A, Z, is_proton, str(e)))

    return samples, skipped
