import os
os.chdir(os.path.dirname(__file__))
import pandas as pd
import numpy as np
from neuroCombat import neuroCombat, neuroCombatFromTraining
import yaml
import pickle as pkl
from scipy.stats import anderson_ksamp
import itertools
import ast   
import warnings
import sys
warnings.filterwarnings("ignore", '.*Consider specifying `method`')

class NestedComBat:

    def load_data(self, radiomics_file, metadata_file):
        radiomics_data = pd.read_csv(radiomics_file)
        metadata_data = pd.read_csv(metadata_file)
        return radiomics_data, metadata_data
    
    def encode_metadata(self, metadata):
        #Encode slice thickness information
        for i, val in enumerate(metadata['Slice_Thickness']):
            if (val >= 0  and val <= 1):
                metadata.loc[i, 'Slice_Thickness'] = 1  
            elif (val > 1  and val <= 2):
                metadata.loc[i, 'Slice_Thickness'] = 2
            elif (val > 2  and val <= 3):
                metadata.loc[i, 'Slice_Thickness'] = 3
            elif val > 3:
                metadata.loc[i, 'Slice_Thickness'] = 4
            else:
                metadata.loc[i, 'Slice_Thickness'] = np.nan
                print(f'Warning... Slice thickness = {val}mm - {metadata.loc[i, "PatientID"]} - {metadata.loc[i, "date"]}') 
                
        #Categorize tube voltage
        for i, val in enumerate(metadata['KVP']):
            if (val <= 100):
                metadata.loc[i, 'KVP'] = 1 
            elif (val > 100 ):
                metadata.loc[i, 'KVP'] = 2
            else:
                metadata.loc[i, 'KVP'] = np.nan
                print(f'Warning... KVP = {val} - {metadata.loc[i, "PatientID"]} - {metadata.loc[i, "date"]}') 
        
        return metadata

    def anderson_darling_test(self, data, batch):
        features_columns = [c for c in data.columns.values if c != batch]
        ad_test = pd.DataFrame(index = range(1), columns = features_columns)
        #get all colummns name except batch
        #Extract a list of dataframe where each dataframe is a subset of the data that has the same batch variable
        list_vector = list()
        list_vector.append([data[(data[batch] == i).values] for i in np.unique(data[batch])])
        list_vector = list_vector[0]

        ## #Anderson-Darling test to prove that the distributions of features with different batch effect are different
        for i, col in enumerate(features_columns):
            try:
                ad_test.iloc[0,i] = anderson_ksamp([df.iloc[:,df.columns.get_loc(col)] for df in list_vector]).significance_level
            except:
                ad_test.iloc[0,i] = np.nan
        return ad_test

    def calculate_andersondarling_differences(self, ad_df):
        categories = ['batch-independent (p > 0.05)','batch-dependent (p <= 0.05)']
        df = pd.DataFrame(columns=categories)
        df.loc[categories[0]] = (ad_df > 0.05).all(axis = 0).sum()/len(ad_df.columns)
        df.loc[categories[1]] = (ad_df <= 0.05).any(axis = 0).sum()/len(ad_df.columns) 
        return df

    def train_combat(self, data, batch_col, bio_col, covars, discard_batch_dependent_features = False, reference_batch = None):
        # Apply ComBat algorithm here
        #Anderson darling test for each feature
        ad_test_preharmo = self.anderson_darling_test(pd.concat([data, covars[batch_col]], axis = 1), batch_col)
        #apply harmonization only to batch-dependent features
        features_names_to_harmo = ad_test_preharmo.loc[:, (ad_test_preharmo < 0.05).values[0]].columns.values
        print(f'Number of batch-dependent features BEFORE HARMO for {batch_col} =', (len(features_names_to_harmo)))
        
        if len(features_names_to_harmo) >0:   
            print('Applying ComBat...')
            #Transpose dataframe to fit combat function
            data_process = data[features_names_to_harmo].T 
            if reference_batch is not None:
                counts = covars[batch_col].value_counts()
                ref_batch = counts.index[counts == counts.max()].values[0]
            else:
                ref_batch = None
            ## COMBAT harmonization
            print(f'TRAIN COMBAT - batch effect {batch_col}')
            
            data_combat = neuroCombat(dat=data_process,
                covars=covars,
                batch_col= batch_col,
                categorical_cols = bio_col,
                eb = True,
                parametric = True,
                ref_batch= ref_batch,
                )
            #Retrive estimates and use them to apply harmonization to the validation set
            estimates = data_combat['estimates']
            data_combat = pd.DataFrame(data_combat['data'].T, columns = features_names_to_harmo)
            
            data[features_names_to_harmo] = data_combat
        else:
            estimates = None
            print('No features to harmonize')
        
        ad_test_postharmo = self.anderson_darling_test(pd.concat([data, covars[batch_col]], axis = 1), batch_col)
        print(f'Number of batch-dependent features AFTER HARMO for {batch_col} = {(ad_test_postharmo < 0.05).sum().sum()} ({round((ad_test_postharmo < 0.05).sum().sum()/len(data.columns.values),3)})')

        if discard_batch_dependent_features:
            print('Removing batch-dependent features...')
            #remove features that are not harmonized
            harmonized_features_name = ad_test_postharmo.loc[:, (ad_test_postharmo >= 0.05).values[0]].columns
            #remove unharmonized features
            data.drop(columns = [f for f in data.columns.values if f not in harmonized_features_name], inplace = True)
            data.reset_index(drop = True, inplace = True)
        else:
            harmonized_features_name = data.columns.values
            
        return data, estimates, features_names_to_harmo, harmonized_features_name

    def train_nested_combat(self, radiomics_data, metadata_data, config_data, output_dir = './output'):
        features_name = radiomics_data.columns.values
        # Get the batch columns and bio covariates
        batch_col = config_data['batch_col']
        covars = metadata_data[batch_col + config_data['bio_col']]
        
        #Define a list with all possible permutation of batch effects
        permutations_batch_effects = list(itertools.permutations(batch_col))
        
        #Define dataframe with the number of batch dependent features due to a particular combination of batch effects
        harmonization_info = pd.DataFrame(index = range(len(permutations_batch_effects)), columns = ['batch_order', 'n_batch_dependent_before_harmo','n_batch_dependent_after_harmo', 'list_batch_dependent_before_harmo', 'list_batch_dependent_after_harmo'])

        #Test for batch dependent features before harmonization
        ad_test_preharmo = pd.DataFrame(index = batch_col, columns = features_name)
        for i, batch_column in enumerate(batch_col):
            ad_test_preharmo.iloc[i] = self.anderson_darling_test(pd.concat([radiomics_data[features_name], covars[batch_column]], axis = 1), batch_column)
            print(f'Number of batch-dependent features BEFORE HARMONIZATION {batch_column} = {len(ad_test_preharmo.loc[batch_column, ad_test_preharmo.loc[batch_column] < 0.05])} ({round(len(ad_test_preharmo.loc[batch_column, ad_test_preharmo.loc[batch_column] < 0.05])/len(features_name),3)})')
        
        #Save all the harmonizard dataframes
        harmonized_features_list = []
        #Save anderson-darling tests
        ad_test_list = []
        #Save estimates 
        estimates_list = []
        #save features to harmonize
        features_to_harmo_list = []
        #Save unharmonized feature list if discard_batch_dependent_features = True
        harmonized_features_name_list = []

        #Loop over each possible combination
        for indx, permutation in enumerate(permutations_batch_effects):
            features_df_harmonized = radiomics_data.copy()
            print(f'PROCESSING PERMUTATION: {[per for per in permutation]}...')
            estimates_permutation = []
            features_to_harmo_permutation = []
            
            try:
                for i, batch_effect in enumerate(permutation):
                    features_df_harmonized, estimates, features_names_to_harmo, harmonized_features_name = self.train_combat(features_df_harmonized, batch_effect, config_data['bio_col'], covars, reference_batch = config_data['reference_batch'], discard_batch_dependent_features = False)
                    estimates_permutation.append(estimates)
                    features_to_harmo_permutation.append(features_names_to_harmo)
                    
                ad_test_train_permutation = pd.DataFrame(index = permutation, columns = features_name)
                for row, batch_effect in enumerate(permutation):   
                    ad_test_train_permutation.iloc[row] = self.anderson_darling_test(pd.concat([features_df_harmonized[features_name], covars[batch_effect]], axis = 1), batch_effect)
                    print(f'Number of batch-dependent features BEFORE COMBAT for {batch_effect} = {len(ad_test_preharmo.loc[batch_effect,(ad_test_preharmo.loc[batch_effect] < 0.05).values].index.values)}')
                    print(f'Number of batch-dependent features AFTER COMBAT for {batch_effect}   = {sum((ad_test_train_permutation.iloc[row] < 0.05))}')
                
                #Append results to lists
                estimates_list.append(estimates_permutation)   
                features_to_harmo_list.append(features_to_harmo_permutation)
                harmonized_features_list.append(features_df_harmonized)
                ad_test_list.append(ad_test_train_permutation)
                
                #Harmonization results of actual permutation
                harmonization_info.loc[indx,'batch_order'] = list(permutation)# type: ignore
                harmonization_info.loc[indx, 'n_batch_dependent_before_harmo'] = sum((ad_test_train_permutation >= 0.05).all(axis = 0))
                harmonization_info.loc[indx, 'n_batch_dependent_after_harmo'] = sum((ad_test_train_permutation < 0.05).any(axis = 0))
                harmonization_info.loc[indx, 'list_batch_dependent_before_harmo'] = list(ad_test_train_permutation.columns[(ad_test_train_permutation >= 0.05).all(axis = 0)])
                harmonization_info.loc[indx, 'list_batch_dependent_after_harmo'] = list(ad_test_train_permutation.columns[(ad_test_train_permutation < 0.05).any(axis = 0)])
                
                if config_data['discard_batch_dependent_features']:
                    harmonized_features_name_list.append(ad_test_train_permutation.columns[(ad_test_train_permutation >= 0.05).all(axis = 0)].values)
            except:
                print(f'Error in permutation: {permutation} - Skipping...')
                continue
        
        ### Get best permutation order
        print('Retriving best order...')
        idx_best_perm = np.where(harmonization_info.n_batch_dependent_after_harmo == harmonization_info.n_batch_dependent_after_harmo.min())[0][0]
        best_harmo_order = harmonization_info.iloc[[idx_best_perm]]
        print(f'Best order: {best_harmo_order["batch_order"]}')
        #Save best results
        os.makedirs(os.path.join(output_dir, 'best_results'), exist_ok=True)
        best_harmo_order.to_csv(os.path.join(output_dir, 'best_results', 'nested_combat_best_order.csv'))
        
        features_df_harmonized_best = harmonized_features_list[idx_best_perm]
        
        if config_data['discard_batch_dependent_features']:
            harmonized_features_name = harmonized_features_name_list[idx_best_perm]
            #remove unharmonized features
            features_df_harmonized_best.drop(columns = [f for f in features_name if f not in harmonized_features_name], inplace = True)
            features_df_harmonized_best.reset_index(drop = True, inplace = True)
        else:
            harmonized_features_name = features_name
        
        #Save estimates
        best_estimates_list = estimates_list[idx_best_perm]
        best_features_to_harmo_list = features_to_harmo_list[idx_best_perm]
        
        #Save estimates to apply them to the test set
        with open(os.path.join(output_dir, 'best_results',f'estimates.pkl'), 'wb') as fp:
                pkl.dump(best_estimates_list, fp)  
        
        with open(os.path.join(output_dir, 'best_results',f'features_to_harmo_list.pkl'), 'wb') as fp:
                pkl.dump(best_features_to_harmo_list, fp)
                
        #Anderson-Darling test of the best order
        ad_test_postharmo_best = ad_test_list[idx_best_perm]
        print(f'Number of batch-dependent features BEFORE NestedCombat harmonization TRAINING = {sum((ad_test_preharmo < 0.05).any(axis = 0))} ({round(sum((ad_test_preharmo < 0.05).any(axis = 0))/len(features_name),3)})')
        print(f'Number of batch-dependent features AFTER NestedCombat harmonization TRAINING = {sum((ad_test_postharmo_best < 0.05).any(axis = 0))} ({round(sum((ad_test_postharmo_best < 0.05).any(axis = 0))/len(features_name),3)})')

        return features_df_harmonized_best

    def apply_nested_combat(self, data, best_col_order, covars, estimates_list, features_to_harmo_list, reference_batch = None):
        print('Applying Nested ComBat harmonization...')
        
        harmonized_data = data.copy()
        for i, batch_effect in enumerate(best_col_order):
            data_process = harmonized_data[features_to_harmo_list[i]].T 
            
            #Get the endoding of the batch effect as in the training set
            if estimates_list[i] is not None and reference_batch:
                covars_col = np.array([np.where(estimates_list[i]['encoding_order'] == element)[0][0] for element in covars[batch_effect]])    
            else:
                covars_col = covars[batch_effect]   
                
            data_combat =  neuroCombatFromTraining(dat=data_process, batch=covars_col, estimates=estimates_list[i])   # I should check here that all batch effects were in the training examples !!!!!!!!!
            data_combat = pd.DataFrame(data_combat['data'].T, columns = features_to_harmo_list[i])
            
            harmonized_data[features_to_harmo_list[i]] = data_combat
            
        return harmonized_data
     
    # Usage example
    def run(self):
        input_dir = '/app/input/'
        output_dir = '/app/output/'
        
        with open(os.path.join(input_dir,'config_file.yaml')) as f:
            config_data = yaml.safe_load(f)
        
        sys.stdout = open(os.path.join(output_dir, f"output_log_{config_data['mode']}.txt"), 'w')
        
        print(f'{config_data["mode"].capitalize()} Nested ComBat...')
        radiomics_data, metadata_data = self.load_data(os.path.join(input_dir, 'radiomics.csv'), os.path.join(input_dir, 'metadata.csv'))
        
        if config_data['mode'] not in ['train', 'test']:
            raise Exception('Invalid mode. Please use train or test.')
        
        #encode metadata
        metadata_data = self.encode_metadata(metadata_data)
        
        #check no missing values in dataframes
        if radiomics_data.isnull().values.any():
            raise Exception('Missing values in radiomics data')
        if metadata_data.isnull().values.any():
            nan_col = metadata_data.columns[metadata_data.isnull().any()].values
            print(f'Warning: Missing values in metadata data. Removing columns {list(nan_col)} with missing values')
            metadata_data.dropna(axis=1, inplace=True)
        
            missing_batch_cols = set(config_data['batch_col']) & set(nan_col)
            if missing_batch_cols:
                raise Exception(f'Batch columns {missing_batch_cols} have missing values. Please remove them before harmonization!')
        
        #copy original radiomics data
        radiomics_data_original = radiomics_data.copy()
        #remove column PatientID
        radiomics_data.drop(columns = 'PatientID', inplace = True)
        
        if config_data['mode'] == 'train':
            harmonized_data = self.train_nested_combat(radiomics_data, metadata_data, config_data, output_dir)
            print('Training completed successfully!')
        elif config_data['mode'] == 'test':
            if os.path.exists(os.path.join(output_dir, 'best_results', 'estimates.pkl')) and os.path.exists(os.path.join(output_dir, 'best_results', 'features_to_harmo_list.pkl')) and os.path.exists(os.path.join(output_dir, 'best_results', 'nested_combat_best_order.csv')):
                with open(os.path.join(output_dir, 'best_results', 'estimates.pkl'), 'rb') as f:
                    estimates_list = pkl.load(f)
                with open(os.path.join(output_dir, 'best_results', 'features_to_harmo_list.pkl'), 'rb') as f:
                    features_to_harmo_list = pkl.load(f)
                #open best order csv
                best_col_order = pd.read_csv(os.path.join(output_dir, 'best_results', 'nested_combat_best_order.csv'))
            else:
                raise Exception('Estimates not found. Please run the training mode first.')
            harmonized_data = self.apply_nested_combat(radiomics_data, ast.literal_eval(best_col_order['batch_order'].values[0]), metadata_data, estimates_list, features_to_harmo_list, config_data['reference_batch'])
            print('Testing completed successfully!')
        else:
            raise Exception('Invalid mode. Please use train or test.')
        
        harmonized_data = pd.concat([radiomics_data_original['PatientID'], harmonized_data], axis = 1)
        
        #save harmonized data
        os.makedirs(output_dir, exist_ok=True)
        harmonized_data.to_csv(os.path.join(output_dir, f'harmonized_radiomics_{config_data["mode"]}.csv'), index=False)
        print(f'Harmonized data saved successfully into directory "{output_dir}"!')

if __name__ == "__main__":
    nested_combat = NestedComBat()
    nested_combat.run()