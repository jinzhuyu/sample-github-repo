You are helping me design a clean, professional GitHub repository for a machine learning / data science project.

Use this repository as a structural and formatting reference:  
https://github.com/jinzhuyu/sample-github-repo/tree/main

My project details:

* Domain: [briefly describe your topic]

* Data: [describe whether you are using existing data, collected data, or synthetic data]

* Task: [e.g., prediction, classification, optimization, simulation, etc.]

* Models / methods: [list methods you plan to use]

Your task is to generate a repository that is **well-structured, reproducible, and easy to understand**, while adapting to my specific project.

* * *

### 1. Repository structure

Create an appropriate folder structure. Only include components that are relevant to my project. For example:

* data/
  
  * raw_data/ (if applicable)
  
  * processed_data/ (if applicable)

* src/ (all code scripts or modules)

* output/
  
  * results (metrics, tables)
  
  * figures (plots, visualizations)

* doc/ (optional documentation, notes)

* README.md

* requirements.txt

If certain components are not needed (e.g., no raw data or no preprocessing), **omit them**.

* * *

### 2. README.md

Write a clear, professional README that includes:

* Project overview (problem, motivation, goal)

* Description of the workflow/pipeline (tailored to this project)

* Explanation of each script/module

* Data description (source, variables, preprocessing if applicable)

* Installation instructions

* Usage instructions (how to run the project)

* Expected outputs (what files/results are generated)

* Notes on assumptions and limitations

Also include a **workflow diagram** (ASCII or markdown) showing the logical flow of the project.

* * *

### 3. Code organization

Generate modular and clean Python script templates adapted to my project. Examples (only include what is relevant):

* data loading / preprocessing script (if needed)

* model training script

* evaluation script

* visualization / plotting script

* main or run_all.py to execute the pipeline (if appropriate)

Requirements:

* Use clear function-based structure (avoid long scripts)

* Use consistent relative paths

* Add concise but informative comments

* Ensure reproducibility (set random seeds where relevant)

* * *

### 4. Modeling and evaluation

Adapt to the project:

* Do NOT assume specific models (choose based on my description)

* Include appropriate evaluation metrics for the task

* Keep implementation simple and educational unless otherwise specified

* * *

### 5. Visualization

Include plotting/visualization scripts if relevant:

* prediction vs observation (for regression)

* confusion matrix (for classification)

* or other task-appropriate visuals

* * *

### 6. Robustness / validation (if applicable)

If relevant, include simple validation or robustness checks such as:

* cross-validation

* sensitivity to input changes

* basic uncertainty analysis

If not applicable, skip this.

* * *

### 7. General principles

* Keep everything **clear, minimal, and reproducible**

* Avoid unnecessary complexity

* Make the repository easy for another student to understand and run

This is for an academic project, so prioritize **clarity, structure, and correctness** over sophistication.
