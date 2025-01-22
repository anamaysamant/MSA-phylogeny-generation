FAMILIES = ["PF00004"]
MSA_TYPES = ["seed"]
CONTEXT_TYPES = ["static","dynamic"]
CONTEXT_SIZES = ["10"]

num_simulations = 10

SIM_INDS = list(range(1,num_simulations+1))
SIM_INDS = list(map(str,SIM_INDS))

rule all:
    input:
        expand("scores/msa-{msa_type}-simulations/MSA-1b/{context_type}-context/{context_size}/{fam}/{fam}-{sim_ind}.tsv",
                msa_type = MSA_TYPES, context_type = CONTEXT_TYPES, context_size = CONTEXT_SIZES, fam = FAMILIES, sim_ind = SIM_INDS),
        expand("scores/msa-{msa_type}-simulations/Potts/{fam}/{fam}-{sim_ind}.tsv",
                msa_type = MSA_TYPES, fam = FAMILIES, sim_ind = SIM_INDS),
        expand("scores/msa-{msa_type}-simulations/ESM2/{fam}/{fam}-{sim_ind}.tsv",
                msa_type = MSA_TYPES, fam = FAMILIES, sim_ind = SIM_INDS)
        
# rule generate_tree:
#     input:
#         "data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta"
#     output:
#         "data/{msa_type}-trees/{fam}_{msa_type}.newick"
#     log:
#         "logs/generate-{msa_type}-trees/{fam}.log"
#     shell:
#         "FastTree {input} > {output}"

rule simulate_along_phylogeny_MSA:
    input:
        MSA="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        tree="data/{msa_type}-trees/{fam}_{msa_type}.newick"
    output:
        "data/msa-{msa_type}-simulations/MSA-1b/{context_type}-context/{context_size}/{fam}/{fam}-{sim_ind}.fasta"
    shell:
        """
        python simulate_along_phylogeny.py --tool MSA_1b --output {output} --input_MSA {input.MSA} --input_tree {input.tree} \
        --context_type {wildcards.context_type} --context_size {wildcards.context_size} --seed {wildcards.sim_ind}
        """

rule simulate_along_phylogeny_MSA_chunked:
    input:
        MSA="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        tree="data/{msa_type}-trees/{fam}_{msa_type}.newick"
    output:
        "data/msa-{msa_type}-simulations/MSA-1b/static-context/chunked/{context_size}/{fam}/{fam}-{sim_ind}.fasta"
    shell:
        """
        python simulate_along_phylogeny.py --output {output} --input_MSA {input.MSA} --input_tree {input.tree} \
        --tool MSA-1b --context_type static --context_size {wildcards.context_size} --chunked --seed {wildcards.sim_ind}
        """
rule simulate_along_phylogeny_Potts:
    input:
        MSA="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        tree="data/{msa_type}-trees/{fam}_{msa_type}.newick",
        J_params="data/protein-families-DCA-params/{fam}_J.npy",
        h_params="data/protein-families-DCA-params/{fam}_h.npy"
    output:
        "data/msa-{msa_type}-simulations/Potts/{fam}/{fam}-{sim_ind}.fasta"
    shell:
        """
        python simulate_along_phylogeny.py --output {output} --input_MSA {input.MSA} --input_tree {input.tree} \
        --J_params {input.J_params} --h_params {input.h_params} --tool Potts --seed {wildcards.sim_ind}
        """
rule simulate_along_phylogeny_ESM2:
    input:
        MSA="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        tree="data/{msa_type}-trees/{fam}_{msa_type}.newick"
    output:
        "data/msa-{msa_type}-simulations/ESM2/{fam}/{fam}-{sim_ind}.fasta"
    shell:
        """
        python simulate_along_phylogeny.py --output {output} --input_MSA {input.MSA} --input_tree {input.tree} \
        --tool ESM2 --seed {wildcards.sim_ind}
        """

rule generate_scores_MSA:
    input:
        original_MSA="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        simulated_MSA="data/msa-{msa_type}-simulations/MSA-1b/{context_type}-context/{context_size}/{fam}/{fam}-{sim_ind}.fasta",
        hmm="data/protein-families-hmms/{fam}.hmm",
        J_params="data/protein-families-DCA-params/{fam}_J.npy",
        h_params="data/protein-families-DCA-params/{fam}_h.npy"
    output:
        ungapped_seq=temp("data/msa-{msa_type}-simulations/MSA-1b/{context_type}-context/{context_size}/{fam}/{fam}-{sim_ind}-ungapped.fasta"),
        hmm_table=temp("scores/msa-{msa_type}-simulations/MSA-1b/{context_type}-context/{context_size}/{fam}/{fam}-{sim_ind}.tbl"),
        score_table="scores/msa-{msa_type}-simulations/MSA-1b/{context_type}-context/{context_size}/{fam}/{fam}-{sim_ind}.tsv"
    shell:
        """
        seqkit replace -s -p "-" -r "" {input.simulated_MSA} > {output.ungapped_seq}
        hmmsearch --tblout {output.hmm_table} {input.hmm} {output.ungapped_seq}  
        python scores_generator.py --input_hmmer {output.hmm_table} --output {output.score_table} --J_params {input.J_params} \
        --h_params {input.h_params} --simulated_MSA {input.simulated_MSA} --original_MSA {input.original_MSA}
        """

rule generate_scores_Potts:
    input:
        original_MSA="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        simulated_MSA="data/msa-{msa_type}-simulations/Potts/{fam}/{fam}-{sim_ind}.fasta",
        hmm="data/protein-families-hmms/{fam}.hmm",
        J_params="data/protein-families-DCA-params/{fam}_J.npy",
        h_params="data/protein-families-DCA-params/{fam}_h.npy"
    output:
        ungapped_seq=temp("data/msa-{msa_type}-simulations/Potts/{fam}/{fam}-{sim_ind}-ungapped.fasta"),
        hmm_table=temp("scores/msa-{msa_type}-simulations/Potts/{fam}/{fam}-{sim_ind}.tbl"),
        score_table="scores/msa-{msa_type}-simulations/Potts/{fam}/{fam}-{sim_ind}.tsv"
    shell:
       """
        seqkit replace -s -p "-" -r "" {input.simulated_MSA} > {output.ungapped_seq}
        hmmsearch --tblout {output.hmm_table} {input.hmm} {output.ungapped_seq}  
        python scores_generator.py --input_hmmer {output.hmm_table} --output {output.score_table} --J_params {input.J_params} \
        --h_params {input.h_params} --simulated_MSA {input.simulated_MSA} --original_MSA {input.original_MSA}
        """

rule generate_scores_ESM2:
    input:
        original_MSA="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        simulated_MSA="data/msa-{msa_type}-simulations/ESM2/{fam}/{fam}-{sim_ind}.fasta",
        hmm="data/protein-families-hmms/{fam}.hmm",
        J_params="data/protein-families-DCA-params/{fam}_J.npy",
        h_params="data/protein-families-DCA-params/{fam}_h.npy"

    output:
        ungapped_seq=temp("data/msa-{msa_type}-simulations/ESM2/{fam}/{fam}-{sim_ind}-ungapped.fasta"),
        hmm_table=temp("scores/msa-{msa_type}-simulations/ESM2/{fam}/{fam}-{sim_ind}.tbl"),
        score_table="scores/msa-{msa_type}-simulations/ESM2/{fam}/{fam}-{sim_ind}.tsv"
    shell:
       """
        seqkit replace -s -p "-" -r "" {input.simulated_MSA} > {output.ungapped_seq}
        hmmsearch --tblout {output.hmm_table} {input.hmm} {output.ungapped_seq}  
        python scores_generator.py --input_hmmer {output.hmm_table} --output {output.score_table} --J_params {input.J_params} \
        --h_params {input.h_params} --simulated_MSA {input.simulated_MSA} --original_MSA {input.original_MSA}
        """
