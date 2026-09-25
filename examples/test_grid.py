from pathlib import Path

import numpy as np
import tdpy


def run_example(output_directory: Path | None = None) -> None:
    """Generate grid diagnostics in the repository-local visuals directory."""

    if output_directory is None:
        output_directory = tdpy.get_visuals_path() / "grid_example"
    output_directory.mkdir(parents=True, exist_ok=True)

    random_generator = np.random.default_rng(0)
    data = random_generator.uniform(0, 1, (100, 5))
    parameter_labels = [[str(index + 1), ""] for index in range(data.shape[1])]

    tdpy.plot_grid(
        np.array(parameter_labels),
        data,
        listlablpopl=["stable"],
        typeplottdim="hist",
        pathbase=f"{output_directory}/",
    )


if __name__ == "__main__":
    run_example()


