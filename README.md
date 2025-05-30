<h1> Genetic Tool to Cell Type Bayesian Mapper (GT2CT Bayesian Mapper) </h1>
<p> 
GT2CT Bayesian Mapper is a Python-based toolkit designed to estimate
the probable fractions of various brain cell types (e.g., class,
subclass, supercluster, or cluster) within GFP/RFP-positive cells
identified by genetic tools. See the image below for an algorithm
schematic.
</p>

![alt text](https://github.com/astronikil/GT2CT/blob/main/images/schematic.png)

<h2> Algorithm </h2>
The technical details of the algorithm are present in <a href="https://github.com/astronikil/GT2CT/blob/main/notes/note.pdf"> this document </a>.

<h2> Getting Started </h2>
<p>To begin, download this code suite to your local machine and navigate into its directory.</p>
<pre><code class="language-bash">git clone https://github.com/astronikil/GT2CT.git
cd GT2CT
</code></pre>
<p>Next, install the necessary packages by creating and populating a Conda virtual environment named <code>GT2CT</code>.</p>
<pre><code class="language-bash">conda env create -p ./GT2CT -f requirements.yml
</code></pre>
<p>After the environment is created, activate it using the following command:</p>
<pre><code class="language-bash">conda activate ./GT2CT
</code></pre>
<p>Finally, you can open the Jupyter Notebook <code>main.ipynb</code>, which contains the algorithm.</p>

<h2> Background </h2>

You can read more about the Allen Brain Atlas <a href =
"https://portal.brain-map.org/atlases-and-data/bkp/abc-atlas">here</a>.
This toolkit is an entry to the data challenge "Map My Sections'.
The details of the challenge are <a href =
"https://alleninstitute.org/events/mapmysections/">here</a>.
