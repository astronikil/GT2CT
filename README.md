<h1> Genetic Tool to Cell Type Bayesian Mapper (GT2CT Bayesian Mapper) </h1>
<p> GT2CT Bayesian Mapper is a python based toolkit
for finding the likely fractions of brain cell-types (class, subclass, supercluster, or cluster) present in
the GFP/RFP positive cells tagged by genetic tools. The schematic of the algorithm is given in the picture below.
</p>

![alt text](https://github.com/astronikil/GT2CT/blob/main/images/schematic.png)

<h2> Algorithm </h2>
The technical details of the algorithm are present in <a href="https://github.com/astronikil/GT2CT/blob/main/notes/note.pdf"> this document </a>.

<h2>Setting Started</h2>
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
