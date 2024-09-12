import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from SPACEL.setting import set_environ_seed
set_environ_seed(42)
from SPACEL import Splane
import scanpy as sc
import matplotlib
import pandas as pd
import numpy as np
import sys
k = int(sys.argv[1])
d_l = float(sys.argv[2])
gnn_dropout = float(sys.argv[3])
c = int(sys.argv[4])

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.serif'] = ['Arial']
sc.settings.set_figure_params(dpi=100,dpi_save=300,facecolor='white',fontsize=10,vector_friendly=True,figsize=(3,3))

'Load ST data and CNV info'
cnv = pd.read_csv('/home/qukun/ccp1997/GBM_10X/ng_revised/cnv_revised/processed_cnv/All_slides_cnvmean_norm.csv',index_col=0)
h5ad_list = os.listdir('/home/qukun/ccp1997/GBM_10X/ng_revised/spatial_h5ad/processed_h5ad')
h5ad_list.sort()

celltypes = [
    'q05cell_abundance_w_sf_AC-like',
    'q05cell_abundance_w_sf_B cells',
    'q05cell_abundance_w_sf_CD4T',
    'q05cell_abundance_w_sf_Cytotoxic CD8T',
    'q05cell_abundance_w_sf_DC cells',
    'q05cell_abundance_w_sf_Endo',
    'q05cell_abundance_w_sf_Excitatory neurons',
    'q05cell_abundance_w_sf_Exhausted CD8T',
    'q05cell_abundance_w_sf_Inhibitory neurons',
    'q05cell_abundance_w_sf_MES-like',
    'q05cell_abundance_w_sf_Monocytes',
    'q05cell_abundance_w_sf_NK cells',
    'q05cell_abundance_w_sf_NPC-like',
    'q05cell_abundance_w_sf_OPC-like',
    'q05cell_abundance_w_sf_Oligodendrocyte',
    'q05cell_abundance_w_sf_Pericyte',
    'q05cell_abundance_w_sf_T-HSPA1A',
    'q05cell_abundance_w_sf_TAM1',
    'q05cell_abundance_w_sf_TAM3',
    'q05cell_abundance_w_sf_TAM4',
    'q05cell_abundance_w_sf_TAM5',
    'q05cell_abundance_w_sf_Treg cells',
    'q05cell_abundance_w_sf_VSMCs'
]

st_ad_list = []
st_id_list = []
for f in h5ad_list:
    'Remove severely damaged slice'
    if f in ['GBM_5.h5ad']:
        print('Skip',f)
        continue
    print('Load:',f)
    deconv_res = pd.read_csv('/home/qukun/ccp1997/GBM_10X/ng_revised/c2location/process_c2location_v2/'+f[:-13]+'.csv',index_col=0)
    if f[:-13].startswith('T_'):
        ad = sc.read_h5ad(f'raw_h5ad/Visium_{f[:-13]}_raw.h5ad')
        deconv_res.index = [i.split('-')[0]+'-1' for i in deconv_res.index]
    elif f[:-13] == 'Visium_S1':
        ad = sc.read_h5ad('raw_h5ad/Visium_S1_raw.h5ad')
        deconv_res.index = [i.split('-')[0]+'-1' for i in deconv_res.index]
    else:
        ad = sc.read_h5ad('/home/qukun/ccp1997/GBM_10X/ng_revised/spatial_h5ad/processed_h5ad/'+f)
    
    deconv_res = (deconv_res/deconv_res.values.sum(1,keepdims=True)).fillna(0)
    deconv_res_max = []
    for celltype in deconv_res.columns:
        deconv_res_max.append(np.partition(deconv_res[celltype],kth=int(len(deconv_res[celltype])*0.99))[int(len(deconv_res[celltype])*0.99)])
    deconv_res_max = pd.Series(deconv_res_max,index=deconv_res.columns)
    deconv_res = np.clip(deconv_res,np.zeros(len(deconv_res_max)),np.array(deconv_res_max),axis=1)
    ad.uns['celltypes'] = celltypes+['chr7','chr10']
    ad.obs[deconv_res.columns] = deconv_res.loc[ad.obs_names,:].values
    ad.obs.index = [i.split('-')[0]+'-'+f[:-13] for i in ad.obs.index]
    ad.obs.loc[np.intersect1d(cnv.index,ad.obs.index),['chr7','chr10']] = cnv.loc[np.intersect1d(cnv.index,ad.obs.index),['chr7_cnvmean','chr10_cnvmean']].values
    ad.obs[['chr7','chr10']] = ad.obs[['chr7','chr10']].fillna(0)
    if 'spatial' not in list(ad.obsm.keys()):
        if 'x_array' in ad.obs.columns:
            ad.obsm['spatial'] = ad.obs[['y_array','x_array']].values
        elif 'col' in ad.obs.columns:
            ad.obsm['spatial'] = ad.obs[['col','row']].values
        else:
            raise ValueError('Spatial key not in obs:', ad.obs.columns.tolist())
    ad.obsm['spatial'][:,1] = -ad.obsm['spatial'][:,1]
    st_id_list.append(f[:-13])
    st_ad_list.append(ad)

    
splane = Splane.init_model(st_ad_list, n_clusters=c,k=k,use_weight=True,gnn_dropout=gnn_dropout,simi_neighbor=1)
splane.train(d_l=d_l,simi_l=None)
splane.identify_spatial_domain()

'Traversing splane parameters'
for ad in st_ad_list:
    sc.pl.embedding(ad,basis='spatial',color='spatial_domain',size=35)
for ad,ad_id in zip(st_ad_list,st_id_list):
    if not os.path.exists(f'results/splane_cell2location_v28/k{k}_c{c}_dl{d_l}_dp{gnn_dropout}'):
        os.makedirs(f'results/splane_cell2location_v28/k{k}_c{c}_dl{d_l}_dp{gnn_dropout}')
    ad.obs[['spatial_domain']].to_csv(f'results/splane_cell2location_v28/k{k}_c{c}_dl{d_l}_dp{gnn_dropout}/{ad_id}_spatial_domain.csv')
