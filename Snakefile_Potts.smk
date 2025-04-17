FAMILIES = ["PF00271","PF00004","PF00005"]
MSA_TYPES = ["seed"]
N_MUTATIONS = ["500"]
N_SEQUENCES = ["50"]
INIT_SEQS = ["0"]

num_simulations = 10

SIM_INDS = list(range(1,num_simulations+1))
SIM_INDS = list(map(str,SIM_INDS))

rule all:
    input:
        # expand("data/msa-{msa_type}-simulations/Potts/{fam}/init-seq-{init_seq}/{fam}-{sim_ind}.fasta",
        #         msa_type = MSA_TYPES, fam = FAMILIES, sim_ind = SIM_INDS, init_seq = INIT_SEQS),
        expand("data/msa-{msa_type}-simulation-trees/Potts/{fam}/init-seq-{init_seq}/{fam}-{sim_ind}.newick",
                msa_type = MSA_TYPES, fam = FAMILIES, sim_ind = SIM_INDS, init_seq = INIT_SEQS),
        expand("scores/msa-{msa_type}-sim-trees/Potts/{fam}/init-seq-{init_seq}/{fam}-tree.tsv",
                msa_type = MSA_TYPES, fam = FAMILIES, init_seq = INIT_SEQS),
        expand("scores/msa-{msa_type}-simulations/Potts/{fam}/init-seq-{init_seq}/{fam}-{sim_ind}.tsv",
                msa_type = MSA_TYPES, fam = FAMILIES, sim_ind = SIM_INDS, init_seq = INIT_SEQS),
        # expand("scores/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/Potts/{fam}/init-seq-{init_seq}/{fam}-{sim_ind}.tsv",
        #         msa_type = MSA_TYPES, fam = FAMILIES, sim_ind = SIM_INDS, n_mutations = N_MUTATIONS, n_sequences = N_SEQUENCES, init_seq = INIT_SEQS),
        # expand("scores/protein-families-msa-{msa_type}/{fam}_{msa_type}.tsv",
        #         msa_type = MSA_TYPES, fam = FAMILIES)
        # expand("data/{msa_type}-trees/{fam}_{msa_type}_rooted.newick",
        #         msa_type = MSA_TYPES, fam= FAMILIES)
        
# rule generate_tree:
#     input:
#         "data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta"
#     output:
#         "data/{msa_type}-trees/{fam}_{msa_type}.newick"
#     log:
#         "logs/generate-{msa_type}-trees/{fam}.log"
#     shell:
#         "FastTree {input} > {output}"

rule root_tree:
    input:
        "data/{msa_type}-trees/{fam}_{msa_type}.newick"
    output:
        "data/{msa_type}-trees/{fam}_{msa_type}_rooted.newick"
    shell:
        "python scripts/root_tree.py --input {input} --output {output}"

rule generate_tree_Potts:
    input:
        "data/msa-{msa_type}-simulations/Potts/{fam}/init-seq-{init_seq}/{fam}-{sim_ind}.fasta"
    output:
        "data/msa-{msa_type}-simulation-trees/Potts/{fam}/init-seq-{init_seq}/{fam}-{sim_ind}.newick"
    log:
        "logs/msa-{msa_type}-simulation-trees/Potts/{fam}/init-seq-{init_seq}/{fam}-{sim_ind}.log"
    shell:
        "fasttree {input} > {output}"

rule generate_tree_metrics_Potts:
    input:
        simulated_trees=expand("data/msa-{msa_type}-simulation-trees/Potts/{fam}/init-seq-{init_seq}/{fam}-{sim_ind}.newick",
                sim_ind = SIM_INDS, allow_missing = True), 
        simulated_MSAs=expand("data/msa-{msa_type}-simulations/Potts/{fam}/init-seq-{init_seq}/{fam}-{sim_ind}.fasta",
                sim_ind = SIM_INDS, allow_missing = True),
        seed_MSA="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        seed_tree="data/{msa_type}-trees/{fam}_{msa_type}.newick"     
    output:
        "scores/msa-{msa_type}-sim-trees/Potts/{fam}/init-seq-{init_seq}/{fam}-tree.tsv"
    script:
        "scripts/tree_metrics_generator.py"

rule simulate_along_phylogeny_Potts:
    input:
        MSA="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        tree="data/{msa_type}-trees/{fam}_{msa_type}.newick",
        J_params="data/protein-families-DCA-params/{fam}_J.npy",
        h_params="data/protein-families-DCA-params/{fam}_h.npy"
    output:
        "data/msa-{msa_type}-simulations/Potts/{fam}/init-seq-{init_seq}/{fam}-{sim_ind}.fasta"
    log:
        "logs/msa-{msa_type}-simulations/Potts/{fam}/init-seq-{init_seq}/{fam}-{sim_ind}.log"
    shell:
        """
        python scripts/simulate_along_phylogeny.py --output {output} --input_MSA {input.MSA} --input_tree {input.tree} \
        --J_params {input.J_params} --h_params {input.h_params} --tool Potts --seed {wildcards.sim_ind} --start_seq_index {wildcards.init_seq}
        """

rule generate_scores_Potts:
    input:
        original_MSA_seed="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        original_MSA_full="data/protein-families-msa-full/{fam}.fasta",
        simulated_MSA="data/msa-{msa_type}-simulations/Potts/{fam}/init-seq-{init_seq}/{fam}-{sim_ind}.fasta",
        tree="data/{msa_type}-trees/{fam}_{msa_type}.newick",
        hmm="data/protein-families-hmms/{fam}.hmm",
        J_params="data/protein-families-DCA-params/{fam}_J.npy",
        h_params="data/protein-families-DCA-params/{fam}_h.npy"
    output:
        ungapped_seq=temp("data/msa-{msa_type}-simulations/Potts/{fam}/init-seq-{init_seq}/{fam}-{sim_ind}-ungapped.fasta"),
        hmm_table=temp("scores/msa-{msa_type}-simulations/Potts/{fam}/init-seq-{init_seq}/{fam}-{sim_ind}.tbl"),
        score_table="scores/msa-{msa_type}-simulations/Potts/{fam}/init-seq-{init_seq}/{fam}-{sim_ind}.tsv"
    shell:
       """
        seqkit replace -s -p "-" -r "" {input.simulated_MSA} > {output.ungapped_seq}
        hmmsearch --max --tblout {output.hmm_table} {input.hmm} {output.ungapped_seq}  
        python scripts/scores_generator.py --input_hmmer {output.hmm_table} --output {output.score_table} --J_params {input.J_params} \
        --h_params {input.h_params} --simulated_MSA {input.simulated_MSA} --original_MSA_seed {input.original_MSA_seed} \
        --original_MSA_full {input.original_MSA_full} --tree {input.tree}
        """

rule simulate_without_phylogeny_Potts:
    input:
        MSA="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        J_params="data/protein-families-DCA-params/{fam}_J.npy",
        h_params="data/protein-families-DCA-params/{fam}_h.npy"
    output:
        "data/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/Potts/{fam}/init-seq-{init_seq}/{fam}-{sim_ind}.fasta"
    log:
         "logs/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/Potts/{fam}/init-seq-{init_seq}/{fam}-{sim_ind}.log"       
    shell:
        """
        python scripts/simulate_along_phylogeny.py --tool Potts --output {output} --input_MSA {input.MSA} --seed {wildcards.sim_ind} --no_phylogeny \
        --J_params {input.J_params} --h_params {input.h_params} --n_mutations {wildcards.n_mutations} --n_sequences {wildcards.n_sequences} \
         --start_seq_index {wildcards.init_seq}
        """

rule generate_scores_no_phylogeny_Potts:
    input:
        original_MSA_seed="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        original_MSA_full="data/protein-families-msa-full/{fam}.fasta",
        simulated_MSA="data/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/Potts/{fam}/init-seq-{init_seq}/{fam}-{sim_ind}.fasta",
        tree="data/{msa_type}-trees/{fam}_{msa_type}.newick",
        hmm="data/protein-families-hmms/{fam}.hmm",
        J_params="data/protein-families-DCA-params/{fam}_J.npy",
        h_params="data/protein-families-DCA-params/{fam}_h.npy"
    output:
        ungapped_seq=temp("data/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/Potts/{fam}/init-seq-{init_seq}/{fam}-{sim_ind}-ungapped.fasta"),
        hmm_table=temp("scores/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/Potts/{fam}/init-seq-{init_seq}/{fam}-{sim_ind}.tbl"),
        score_table="scores/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/Potts/{fam}/init-seq-{init_seq}/{fam}-{sim_ind}.tsv"
    shell:
       """
        seqkit replace -s -p "-" -r "" {input.simulated_MSA} > {output.ungapped_seq}
        hmmsearch --max --tblout {output.hmm_table} {input.hmm} {output.ungapped_seq}  
        python scripts/scores_generator.py --input_hmmer {output.hmm_table} --output {output.score_table} --J_params {input.J_params} \
        --h_params {input.h_params} --simulated_MSA {input.simulated_MSA} --original_MSA_seed {input.original_MSA_seed} \
        --original_MSA_full {input.original_MSA_full} --tree {input.tree} --no_phylogeny
        """


rule generate_scores_natural:
    input:
        original_MSA_seed="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        original_MSA_full="data/protein-families-msa-full/{fam}.fasta",
        simulated_MSA="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        tree="data/{msa_type}-trees/{fam}_{msa_type}.newick",
        hmm="data/protein-families-hmms/{fam}.hmm",
        J_params="data/protein-families-DCA-params/{fam}_J.npy",
        h_params="data/protein-families-DCA-params/{fam}_h.npy"

    output:
        ungapped_seq=temp("data/protein-families-msa-{msa_type}/{fam}_{msa_type}-ungapped.fasta"),
        hmm_table=temp("scores/protein-families-msa-{msa_type}/{fam}_{msa_type}.tbl"),
        score_table="scores/protein-families-msa-{msa_type}/{fam}_{msa_type}.tsv"
    shell:
       """
        seqkit replace -s -p "-" -r "" {input.simulated_MSA} > {output.ungapped_seq}
        hmmsearch --tblout {output.hmm_table} {input.hmm} {output.ungapped_seq}  
        python scripts/scores_generator.py --input_hmmer {output.hmm_table} --output {output.score_table} --J_params {input.J_params} \
        --h_params {input.h_params} --simulated_MSA {input.simulated_MSA} --original_MSA_seed {input.original_MSA_seed} \
        --original_MSA_full {input.original_MSA_full} --tree {input.tree}
        """

