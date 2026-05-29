from __future__ import annotations

from pathlib import Path
import sys

from tigramite.independence_tests.parcorr import ParCorr


ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from multi_stage_lpcmci_shared import build_parser, run_multi_stage_lpcmci  # noqa: E402


def main() -> None:
    parser = build_parser(
        "Run two-column LPCMCI with structural input assumptions using the default linear ParCorr CI test."
    )
    args = parser.parse_args()

    cond_ind_test = ParCorr(significance="analytic")
    run_multi_stage_lpcmci(
        args=args,
        cond_ind_test=cond_ind_test,
        ci_test_label="parcorr_linear",
        ci_test_summary="ParCorr(significance='analytic')",
        script_path=Path(__file__),
    )


if __name__ == "__main__":
    main()
