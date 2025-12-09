import os

from eks.command_line_args import handle_io, handle_parse_args
from eks.singlecam_smoother import fit_eks_singlecam
from eks.utils import plot_results

def run_smoother(
    input_source: str | list[str],
    smoother_type: str = "singlecam",
    save_dir: str | None = None,
    save_filename: str = "eks_singlecam.csv",
    bodypart_list: list[str] | None = None,
    s: float | list[float] | None = None,
    s_frames: list[tuple[int, int]] | None = None,
    blocks: list[list[int]] = [],
    verbose: bool = False,
):

    # Determine input directory from either a directory or file list
    if isinstance(input_source, str):
        input_dir = os.path.abspath(input_source)
    else:
        input_dir = os.path.abspath(os.path.dirname(input_source[0]))

    # Set up the output directory
    save_dir = handle_io(input_dir, save_dir)

    # Fit the smoother
    output_df, s_finals, input_dfs, bodypart_list = fit_eks_singlecam(
        input_source=input_source,
        save_file=os.path.join(save_dir, save_filename),
        bodypart_list=bodypart_list,
        smooth_param=s,
        s_frames=s_frames,
        blocks=blocks,
        verbose=verbose,
    )

    # Plot last keypoint result
    keypoint_i = -1
    plot_results(
        output_df=output_df,
        input_dfs_list=input_dfs,
        key=f"{bodypart_list[keypoint_i]}",
        idxs=(0, 500),
        s_final=s_finals[keypoint_i],
        nll_values=None,
        save_dir=save_dir,
        smoother_type=smoother_type,
    )

    return output_df, s_finals, bodypart_list, save_dir


if __name__ == "__main__":

    smoother_type = "singlecam"
    args = handle_parse_args(smoother_type)

    # Normalize input source
    input_source = args.input_dir if isinstance(args.input_dir, str) else args.input_files

    run_smoother(
        input_source = input_source,
        smoother_type = smoother_type,
        save_dir = args.save_dir,
        save_filename = args.save_filename,
        bodypart_list = args.bodypart_list,
        s = args.s,
        s_frames = args.s_frames,
        blocks = args.blocks,
        verbose = (True if args.verbose == "True" else False),
    )
