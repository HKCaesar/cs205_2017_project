import kmeans.mpi_kmeans
import kmeans.mpi_pycuda_kmeans
import kmeans.pycuda_kmeans
import kmeans.sequential_kmeans
import kmeans.stock_kmeans


def generate_sbatch_from_template(template_file,insert):

    fp = open(template_file)
    lines = [line for line in fp]

    sbatch = list(filter(lambda s: s.startswith('#SBATCH'), lines))
    sbatch = "".join(sbatch)

    modules = list(filter(lambda s: s.startswith('module load')|s.startswith('source activate'), lines))
    modules = "".join(modules)

    return "\n".join([sbatch, modules, insert])

print(generate_sbatch_from_template('blank.slurm',"python kmeansrunner kmeans"))