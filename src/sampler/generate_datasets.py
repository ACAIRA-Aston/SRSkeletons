import os
import config
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import trange
import matplotlib.pyplot as plt
from src.sampler.sample import generate_data


warnings.filterwarnings("ignore", category=RuntimeWarning)


class GenerateData:
    def __init__(self, difficulty=None, batch_size=8, Ns=None):
        """
        :param difficulty: String ('easy', 'medium', 'hard') or None to load the entire dataset
        :param batch_size: Determines how many skeletons conform a batch
        :param Ns: Number of curves/sets to generate per skeleton
        """
        # Load the specified split if provided, otherwise fallback to the full dataset
        if difficulty in ['easy', 'medium', 'hard']:
            self.skeletons = pd.read_parquet(f'../data/canonical_skeletons_{difficulty}.parquet')
            print(f"Loaded {self.skeletons.shape[0]} skeletons from the '{difficulty}' split.")
        else:
            self.skeletons = pd.read_parquet('../data/canonical_skeletons.parquet')
            print(f"Loaded {self.skeletons.shape[0]} skeletons from the main dataset.")

        self.items_list = np.arange(self.skeletons.shape[0])
        self.batch_size = batch_size
        self.batch_list = [self.items_list[i:i + self.batch_size] for i in range(0, len(self.items_list), self.batch_size)]
        self.Ns = Ns
        if Ns is None:
            self.Ns = config.number_of_sets
        self.n = config.max_number_of_points

    def shuffle(self):
        np.random.shuffle(self.items_list)
        self.batch_list = [self.items_list[i:i + self.batch_size] for i in range(0, len(self.items_list), self.batch_size)]

    def generate_batch(self, batch_id, plot=False):
        batch_indices = self.batch_list[batch_id]
        X, Y = np.zeros((len(batch_indices), self.n, self.Ns)), np.zeros((len(batch_indices), self.n, self.Ns))
        eqs_batch = []
        good_skeleton_indices = []
        n_skeletons = len(self.skeletons)
        max_attempts = 50

        for sk in range(0, self.batch_size):
            print(sk)
            original_idx = batch_indices[sk]
            result = generate_data(self.skeletons.iloc[batch_indices[sk]].skeleton)

            if result is None:
                print(f"I skipped skeleton {self.skeletons.iloc[original_idx].skeleton}, " f"substituting a random skeleton")
                result = None
                attempts = 0

                while result is None and attempts < max_attempts:
                    if good_skeleton_indices:  # Prefer a known-good skeleton if we have one
                        fallback_idx = random.choice(good_skeleton_indices)
                    else:
                        # Pool empty (e.g. very first attempt ever). Sample from the entire skeleton set
                        fallback_idx = random.randrange(n_skeletons)

                    result = generate_data(self.skeletons.iloc[fallback_idx].skeleton)
                    attempts += 1

                    if result is not None:
                        good_skeleton_indices.append(fallback_idx)

                if result is None:
                    raise RuntimeError(f"Could not find a valid substitute skeleton after {max_attempts} attempts.")

            else:
                good_skeleton_indices.append(original_idx)  # This skeleton worked, remember it for future substitutions.

            X[sk, :, :], Y[sk, :, :], eqs = result
            eqs_batch.append(eqs)

            if plot:
                n_cols = int(np.ceil(self.Ns / 2))
                fig, axes = plt.subplots(nrows=2, ncols=n_cols, figsize=(4 * n_cols, 6))
                axes_flat = axes.flatten()
                for i in range(self.Ns):
                    ax = axes_flat[i]
                    ax.scatter(X[sk, :, i], Y[sk, :, i], alpha=0.6, edgecolors='none', s=10)
                    ax.set_title(f'Set {i + 1}')
                    ax.grid(True, linestyle='--', alpha=0.5)
                    ax.autoscale(enable=True, axis='both', tight=False)
                plt.title(self.skeletons.iloc[batch_indices[sk]].skeleton)
                plt.tight_layout()
                plt.show()
                pass

        return X, Y, eqs_batch

    def generate_all_batches(self):
        full_data = []
        for batch_id in trange(0, len(self.batch_list)):
            full_data.append(self.generate_batch(batch_id))
        return full_data

    def curate_dataset(self, difficulty, output_path=None, row_idx=None):
        """
        Interactively loops through the skeletons of the specified difficulty.
        Generates 10 curves for each, plots them in a grid, and prompts you
        via CLI to select the best one to keep. Saves progress incrementally.
        """

        # Load the corresponding difficulty split
        self.skeletons = pd.read_parquet(f'../data/canonical_skeletons_{difficulty}.parquet')
        print(f"\n=======================================================")
        print(f"  Starting Interactive Curation: {difficulty.upper()} Split  ")
        print(f"=======================================================")

        if output_path is None:
            output_path = f"../data/curated_datasets_{difficulty}.parquet"

        # Load existing progress if you are resuming a session
        curated_records = []
        if os.path.exists(output_path):
            try:
                existing_df = pd.read_parquet(output_path)
                curated_records = existing_df.to_dict('records')
                print(f"-> Found existing file with {len(curated_records)} entries. Resuming...")
            except Exception:
                print("-> Could not read existing file. Starting a fresh curation session.")

        # Temporarily force the generator to output exactly 10 sets
        orig_num_sets = getattr(config, 'number_of_sets', 10)
        config.number_of_sets = 10

        # Determine which rows to iterate over
        if row_idx is not None:
            if not (0 <= row_idx < len(self.skeletons)):
                raise IndexError(f"Index {row_idx} is out of bounds for skeletons of length {len(self.skeletons)}.")
            target_skeletons = [(row_idx, self.skeletons.iloc[row_idx])]
            print(f"-> Mode: Single-index curation for row index {row_idx}")
        else:
            target_skeletons = list(self.skeletons.iterrows())

        # Track which skeletons we've already curated
        completed_skeletons = {r['skeleton'] for r in curated_records}

        try:
            for idx, row in target_skeletons:
                skeleton = row['skeleton']
                if row_idx is None and skeleton in completed_skeletons:
                    continue

                print(f"\n[{len(curated_records) + 1}/{len(self.skeletons)}] Curating Skeleton: {skeleton}")

                selected_idx = None
                while selected_idx is None:
                    # Generate the 10 sets of curves
                    result = generate_data(skeleton)
                    if result is None:
                        print("⚠ Generation returned None. Skipping this skeleton.")
                        break

                    X_vals, Y_vals, eqs = result  # shapes: (n_points, 10), (n_points, 10), list of 10 equations

                    # Set up the 2x5 selection grid
                    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(18, 7))
                    axes_flat = axes.flatten()

                    for i in range(10):
                        ax = axes_flat[i]
                        ax.scatter(X_vals[:, i], Y_vals[:, i], alpha=0.7, edgecolors='none', s=8, color='tab:blue')
                        ax.set_title(f"Curve #{i + 1}", fontsize=10, fontweight='bold')
                        ax.grid(True, linestyle='--', alpha=0.5)
                        ax.autoscale(enable=True, axis='both', tight=False)

                    plt.suptitle(f"Skeleton: {skeleton}", fontsize=12, y=0.98)
                    plt.tight_layout()

                    # Draw the plot in a non-blocking window so you can view it while typing in the terminal
                    plt.show(block=False)
                    plt.pause(0.1)  # Forces the GUI backend to render immediately

                    # CLI Prompt
                    user_input = input(
                        "-> Enter curve number to keep (1-10), 'r' to regenerate, or 's' to skip: "
                    ).strip().lower()

                    if user_input == 'r':
                        print("Regenerating 10 new curves for this skeleton...")
                        plt.close(fig)
                        continue
                    elif user_input == 's':
                        print("Skipping skeleton.")
                        plt.close(fig)
                        break

                    try:
                        choice = int(user_input)
                        if 1 <= choice <= 10:
                            selected_idx = choice - 1
                        else:
                            print("❌ Invalid selection. Must be between 1 and 10.")
                    except ValueError:
                        print("❌ Invalid command. Enter 1-10, 'r' to regenerate, or 's' to skip.")

                    plt.close(fig)

                # Save choice immediately to parquet so you never lose progress
                if selected_idx is not None:
                    record = {
                        'skeleton': skeleton,
                        'equation': str(eqs[selected_idx]),
                        'X': X_vals[:, selected_idx].tolist(),
                        # Nested list storage is perfectly handled by Arrow/Parquet
                        'Y': Y_vals[:, selected_idx].tolist()
                    }
                    existing_entry_idx = next((i for i, r in enumerate(curated_records) if r['skeleton'] == skeleton), None)
                    if existing_entry_idx is not None:
                        curated_records[existing_entry_idx] = record
                        print(f"✓ Updated existing record for skeleton at index {row_idx}!")
                    else:
                        curated_records.append(record)
                        print(f"✓ Progress saved! Total curated: {len(curated_records)}/{len(self.skeletons)}")

                    df_to_save = pd.DataFrame(curated_records)
                    df_to_save.to_parquet(output_path, engine='pyarrow', compression='snappy')

        finally:
            # Restore your original configuration rules when the loop exits
            config.number_of_sets = orig_num_sets
            print(f"\nSession terminated. Final progress safely written to: {output_path}")


if __name__ == '__main__':
    generator = GenerateData()
    # generator.clean()
    # b0 = generator.generate_batch(2)
    # generator.generate_all_batches()
    generator.curate_dataset(difficulty='medium')


