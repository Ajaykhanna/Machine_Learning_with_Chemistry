Please analyze the attached PDF file: "lemkul-2024-introductory-tutorials-for-simulating-protein-dynamics-with-gromacs.pdf".

Your goal is to create a comprehensive, web-based tutorial suitable for hosting on GitHub Pages, designed for undergraduate and graduate students learning GROMACS. The tutorial content, including all GROMACS commands, parameters (.mdp file contents), explanations, workflow steps, and cited references, must be extracted *exclusively* from the provided PDF document. Do not add information or steps not present in the PDF.

The final output should consist of three separate files: `index.html`, `styles.css`, and `script.js`.

**Requirements:**

1.  **Content Extraction (from PDF):**
    * Accurately extract the specific GROMACS tutorial steps detailed in the PDF, covering exercises like single protein simulation, protein complex setup, and umbrella sampling (if detailed).
    * Include all necessary GROMACS commands, ensuring they are correctly formatted for a bash terminal.
    * Extract the exact content of any `.mdp` parameter files shown in the PDF.
    * Capture the explanations provided in the PDF for each step and command.
    * List any references mentioned in the tutorial sections of the PDF.

2.  **Structure (`index.html`):**
    * Organize the tutorial into logical sections based on a standard GROMACS workflow, similar to: Introduction, System Setup & Preparation (Structure, Topology, Box/Solvation, Ions), Energy Minimization, Equilibration (NVT, NPT), Production MD, Analysis (PBC correction, RMSD, RMSF, and any specific analyses from the PDF like umbrella sampling), Visualization, and References.
    * Use semantic HTML5 tags (`h1`, `h2`, `h3`, `p`, `ul`, `li`, `section`, `pre`, `code`, `strong`, `em`, `a`).
    * Include GitHub-compatible emojis in section headers for visual appeal (e.g., 📝, ⚙️, 🔥, 🌡️, ▶️, 📊, ✨, 📚).

3.  **Styling (`index.html` + `styles.css`):**
    * Integrate Tailwind CSS via CDN in `index.html`.
    * Apply Tailwind utility classes extensively in `index.html` for layout, typography, colors, backgrounds, padding, margins, borders, rounded corners, and shadows to create a clean, modern look.
    * Link to an external `styles.css` file from `index.html`.
    * In `styles.css`, include custom CSS rules for:
        * Toggle arrows on section headers (`.toggle-header::after`, changing based on a `.collapsed` class).
        * Language hints for code blocks (`pre[data-lang='bash']::before`, `pre[data-lang='ini']::before`).
        * Any minor base style overrides if necessary (e.g., body font).

4.  **Code Formatting (`index.html`):**
    * Display all terminal commands and `.mdp` file content within `<pre><code data-lang="..."></code></pre>` blocks. Use `data-lang="bash"` for commands and `data-lang="ini"` (or similar) for `.mdp` files.
    * Apply appropriate Tailwind classes to code blocks for background color (e.g., dark gray), text color (e.g., white), padding, rounded corners, and horizontal scrolling (`overflow-x-auto`).
    * Ensure code lines are reasonably formatted for readability, ideally keeping commands under ~75 columns where feasible using line breaks (`\`) if appropriate for bash.

5.  **Interactivity (`index.html` + `script.js`):**
    * Make tutorial subsections (e.g., within Setup, Minimization, Equilibration, Analysis) collapsible.
    * Use clickable `<h3>` headers (`.toggle-header`) with an `onclick` attribute calling a JavaScript function.
    * Link to an external `script.js` file from `index.html` (use `defer`).
    * In `script.js`, implement the `toggleVisibility('elementId')` function to toggle Tailwind's `hidden` class on the target content `div` and a `collapsed` class on the clicked header.
    * Include a `DOMContentLoaded` event listener in `script.js` to ensure sections marked initially as `.collapsed` in the HTML start hidden.

6.  **Output Format:**
    * Provide the complete content for `index.html`, `styles.css`, and `script.js` in three separate, clearly labeled code blocks.

Ensure the generated tutorial accurately reflects the procedures and details outlined *only* in the provided PDF document.
