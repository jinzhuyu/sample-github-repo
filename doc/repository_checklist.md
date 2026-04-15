# Repository Submission Checklist

Use this checklist before sharing or submitting your repository.

## Organization
- The repository title is clear and descriptive.
- The top-level folders are logically organized.
- File names are professional and meaningful.
- Raw data and processed data are separated.

## Documentation
- The main `README.md` explains the project clearly.
- The `README.md` includes setup and run instructions.
- The `data/README.md` explains what each dataset contains.
- The project report or presentation is stored in `docs/report/`.

## Code Quality
- The code is split across multiple scripts rather than one giant file.
- Repeated logic is placed into functions.
- Hard-coded personal file paths are avoided.
- Random seeds are fixed where reproducibility matters.

## Reproducibility
- Dependencies are listed in `requirements.txt`.
- The pipeline can be run from the repository root.
- Outputs are generated into the `outputs/` folder.
- Unnecessary intermediate or temporary files are removed.

## Version Control
- Commits are regular and meaningful.
- Commit messages describe specific changes.
- The repository history shows incremental progress.
