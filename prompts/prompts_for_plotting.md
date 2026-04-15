# Role: You are an expert Scientific Visualization Engineer specializing in Python and matplotlib

## Task: Write a Python script to parse molecular dynamics transition dipole moment data and generate publication-ready plots

## Data Specification

The input file is a space-separated or CSV file with the following columns:

   1. Time (fs)
   2. Excited State Label (e.g., 1, 2, 3...)
   3. u_x, 4. u_y, 5. u_z (Transition dipole components)
   4. $|u_{total}^2|$ (Sum of squared components)

Script Requirements:

   1. Data Handling:

* Use pandas to load the file.
  * Group the data by the Excited State Label to plot each state as a separate line.
  * The X-axis should use the Snapshot Index (row index per state) rather than raw time.

   1. Visual Styling:

* Font: Set the global default font to 'Arial'.
  * Colors: Use ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]. For more than 5 states, use the 'tab10' colormap to maintain a consistent visual profile.
  * Axes:
  * Exactly 5 ticks for both X and Y axes.
    * Tick labels must have one decimal point.
    * X Label: "Snapshot Index"
    * Y Label: "$|u_{total}^2|$ [atomic units]" (Use LaTeX rendering).
  * Legend: Position in the Upper Right.

   1. Export Settings:

* Save as a 300 DPI PNG and a vector-based PDF.
  * Use plt.tight_layout() and bbox_inches='tight'.

   1. Code Structure:

* Create a function process_and_plot(file_path) that performs the grouping and plotting.
  * Include a block to generate a synthetic CSV file matching the 6-column structure described so the script is immediately testable.
