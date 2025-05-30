<h1> Genetic Tool to Cell Type Bayesian Mapper (GT2CT Bayesian Mapper) </h1>
<p> GT2CT Bayesian Mapper is a python based toolkit
for finding the likely fractions of brain cell-types (class, subclass, supercluster, or cluster) present in
the GFP/RFP positive cells tagged by genetic tools. The schematic of the algorithm is given in the picture below.
</p>

![alt text](https://github.com/astronikil/GT2CT/blob/main/images/schematic.png)

<h2> Algorithm </h2>
The technical details of the algorithm are present in <a href="https://github.com/astronikil/GT2CT/blob/main/notes/note.pdf"> this document </a>.

<h2> Setting started </h2>

<p> Download this code suite to your local machine. Using terminal, move to the code's folder. It is recommended that 
conda virtual environment <code>GT2CT</code> created and used for the installation of required packages. To do this, 
execute the following command</p>

<code> conda env create -p ./GT2CT -f requirements.yml </code> <br>

Once this is done, activate the virtual environment as 

<code> conda activate ./GT2CT </code> <br>

Then, you can use <code> jupyter notebook</code> and open the main analysis
notebook <code>main.ipynb</code>.


