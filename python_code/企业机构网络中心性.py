import os
import pandas as pd
from tqdm import tqdm
import warnings
import networkx as nx
from scipy.sparse import csr_matrix
warnings.filterwarnings("ignore")
path = 'C:\\Download'
os.chdir(path)

data = pd.read_stata('data.dta')
results_df = pd.DataFrame()
for year in tqdm(data['year'].unique(), desc='Processing each year'):
    data_year = data[data['year'] == year]
    investors = data_year['ID'].unique()
    firms = data_year['Stkcd'].unique()
    S = csr_matrix((len(investors), len(firms)), dtype=int)
    investor_index = {inv: idx for idx, inv in enumerate(investors)}
    firm_index = {f: idx for idx, f in enumerate(firms)}
    for _, row in data_year.iterrows():
        investor_idx = investor_index[row['ID']]
        firm_idx = firm_index[row['Stkcd']]
        S[investor_idx, firm_idx] = 1
    W = S.dot(S.transpose())
    W.setdiag(0)
    G = nx.from_scipy_sparse_array(W)
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    year_results = pd.DataFrame({
        'ID': investors,
        'year': year,
        'degree_centrality': [degree_centrality.get(i) for i in range(len(investors))],
        'betweenness_centrality': [betweenness_centrality.get(i) for i in range(len(investors))],
        'eigenvector_centrality': [eigenvector_centrality.get(i) for i in range(len(investors))]
    })
    results_df = pd.concat([results_df, year_results], ignore_index=True)
results_df = results_df.merge(data[['ID', 'Stkcd', 'year', 'Top1']].drop_duplicates(), on=['ID', 'year'])

final_results = results_df[results_df['Top1'] == 1]
final_results.to_csv('centrality.csv', index=False)