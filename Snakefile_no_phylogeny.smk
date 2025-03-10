FAMILIES = ["PF00004"]
MSA_TYPES = ["seed"]
PROPOSAL_TYPES = ["msa_prob_dist","random"]
CONTEXT_SIZES = ["10"]
N_MUTATIONS = ["500"]
N_SEQUENCES = ["50"]
INIT_SEQS = ["-1"]

num_simulations = 10

SIM_INDS = list(range(1,num_simulations+1))
SIM_INDS = list(map(str,SIM_INDS))

rule all:
    input:
        # expand("scores/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/context-size-{context_size}/{fam}/{fam}-{sim_ind}.tsv",
        #           msa_type = MSA_TYPES, fam = FAMILIES, sim_ind = SIM_INDS, proposal_type = PROPOSAL_TYPES, n_mutations = N_MUTATIONS, 
        #           context_size = CONTEXT_SIZES, n_sequences = N_SEQUENCES, init_seq = INIT_SEQS),
        # expand("scores/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/MSA-1b-finetuned-{fam}/init-seq-{init_seq}/{proposal_type}-proposal/context-size-{context_size}/{fam}/{fam}-{sim_ind}.tsv",
        #         msa_type = MSA_TYPES, fam = FAMILIES, sim_ind = SIM_INDS, proposal_type = PROPOSAL_TYPES, n_mutations = N_MUTATIONS, 
        #         context_size = CONTEXT_SIZES, n_sequences = N_SEQUENCES, init_seq = INIT_SEQS),


rule root_tree:
    input:
        "data/{msa_type}-trees/{fam}_{msa_type}.newick"
    output:
        "data/{msa_type}-trees/{fam}_{msa_type}_rooted.newick"
    shell:
        "python scripts/root_tree.py --input {input} --output {output}"

rule simulate_without_phylogeny_MSA:
    input:
        MSA="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
    output:
        "data/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/context-size-{context_size}/{fam}/{fam}-{sim_ind}.fasta"
    log:
        "logs/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/context-size-{context_size}/{fam}/{fam}-{sim_ind}.log"
    shell:
        """
        python scripts/simulate_along_phylogeny.py --tool MSA_1b --output {output} --input_MSA {input.MSA} --context_size {wildcards.context_size}  \
        --proposal_type {wildcards.proposal_type} --seed {wildcards.sim_ind} --no_phylogeny --n_mutations {wildcards.n_mutations} \
         --n_sequences {wildcards.n_sequences} --start_seq_index {wildcards.init_seq}
        """

rule generate_scores_no_phylogeny_MSA:
    input:
        original_MSA_seed="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        original_MSA_full="data/protein-families-msa-full/{fam}.fasta",
        simulated_MSA="data/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/context-size-{context_size}/{fam}/{fam}-{sim_ind}.fasta",
        tree="data/{msa_type}-trees/{fam}_{msa_type}.newick",
        hmm="data/protein-families-hmms/{fam}.hmm",
        J_params="data/protein-families-DCA-params/{fam}_J.npy",
        h_params="data/protein-families-DCA-params/{fam}_h.npy"

    output:
        ungapped_seq=temp("data/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/context-size-{context_size}/{fam}/{fam}-{sim_ind}-ungapped.fasta"),
        hmm_table=temp("scores/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/context-size-{context_size}/{fam}/{fam}-{sim_ind}.tbl"),
        score_table="scores/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/MSA-1b/init-seq-{init_seq}/{proposal_type}-proposal/context-size-{context_size}/{fam}/{fam}-{sim_ind}.tsv"
    shell:
       """
        seqkit replace -s -p "-" -r "" {input.simulated_MSA} > {output.ungapped_seq}
        hmmsearch --max --tblout {output.hmm_table} {input.hmm} {output.ungapped_seq}  
        python scripts/scores_generator.py --input_hmmer {output.hmm_table} --output {output.score_table} --J_params {input.J_params} \
        --h_params {input.h_params} --simulated_MSA {input.simulated_MSA} --original_MSA_seed {input.original_MSA_seed} \
        --original_MSA_full {input.original_MSA_full} --tree {input.tree} --no_phylogeny
        """

rule simulate_without_phylogeny_MSA_finetuned:
    input:
        MSA="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
    output:
        "data/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/MSA-1b-finetuned-{fam}/init-seq-{init_seq}/{proposal_type}-proposal/context-size-{context_size}/{fam}/{fam}-{sim_ind}.fasta"
    log:
        "logs/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/MSA-1b-finetuned-{fam}/init-seq-{init_seq}/{proposal_type}-proposal/context-size-{context_size}/{fam}/{fam}-{sim_ind}.log"
    shell:
        """
        python scripts/simulate_along_phylogeny.py --tool MSA_1b --output {output} --input_MSA {input.MSA} --context_size {wildcards.context_size}  \
        --proposal_type {wildcards.proposal_type} --seed {wildcards.sim_ind} --no_phylogeny --n_mutations {wildcards.n_mutations} \
        --n_sequences {wildcards.n_sequences} --FT_fam {wildcards.fam} --start_seq_index {wildcards.init_seq}
        """

rule generate_scores_no_phylogeny_MSA_finetuned:
    input:
        original_MSA_seed="data/protein-families-msa-{msa_type}/{fam}_{msa_type}.fasta",
        original_MSA_full="data/protein-families-msa-full/{fam}.fasta",
        simulated_MSA="data/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/MSA-1b-finetuned-{fam}/init-seq-{init_seq}/{proposal_type}-proposal/context-size-{context_size}/{fam}-{sim_ind}.fasta",
        tree="data/{msa_type}-trees/{fam}_{msa_type}.newick",
        hmm="data/protein-families-hmms/{fam}.hmm",
        J_params="data/protein-families-DCA-params/{fam}_J.npy",
        h_params="data/protein-families-DCA-params/{fam}_h.npy"

    output:
        ungapped_seq=temp("data/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/MSA-1b-finetuned-{fam}/init-seq-{init_seq}/{proposal_type}-proposal/context-size-{context_size}/{fam}-{sim_ind}-ungapped.fasta"),
        hmm_table=temp("scores/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/MSA-1b-finetuned-{fam}/init-seq-{init_seq}/{proposal_type}-proposal/context-size-{context_size}/{fam}-{sim_ind}.tbl"),
        score_table="scores/no-phylogeny/{n_mutations}-mutations/{n_sequences}-sequences/msa-{msa_type}-simulations/MSA-1b-finetuned-{fam}/init-seq-{init_seq}/{proposal_type}-proposal/context-size-{context_size}/{fam}-{sim_ind}.tsv"
    shell:
        """
        seqkit replace -s -p "-" -r "" {input.simulated_MSA} > {output.ungapped_seq}
        hmmsearch --max --tblout {output.hmm_table} {input.hmm} {output.ungapped_seq}  
        python scripts/scores_generator.py --input_hmmer {output.hmm_table} --output {output.score_table} --J_params {input.J_params} \
        --h_params {input.h_params} --simulated_MSA {input.simulated_MSA} --original_MSA_seed {input.original_MSA_seed} \
        --original_MSA_full {input.original_MSA_full} --tree {input.tree} --no_phylogeny
            """