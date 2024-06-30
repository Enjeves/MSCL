import torch
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


from sklearn.model_selection import StratifiedKFold


from mmsc.dataset import TabularDataset

from mmsc.scloss import SupConLoss
from mmsc.model import MMSCARF
from mmsc.utils import dataset_embeddings, fix_seed, train_epoch, evaluate_model,change_threshold


def kfold_supmmscarf_svm(Alldataset, target, filepath, importanceFilepath,
                    batch_size = 256,
                    embeddim_poj = 8,
                    epochs = 1000, encoder_depth = 4, head_depth = 2,
                    dropout = 0, dropout_poj = 0,
                    feature_sparsity = 0.3, learnable_sparsity = True,
                    randomseed = 333, lr = 1e-3,
                    temperature = 0.07, base_temperature = 0.07,
                    contrast_mode = 'all',
                    hidden_layer_sizes=(437,200), activation='relu',
                    solver='adam', alpha=0.0001, max_iter=50,
                    repeat = 1):

    
    
    otfile = open(filepath,'w')
    seed = randomseed
    fix_seed(seed)
    print("编码网络不改变特征维度", file=otfile)
    print("编码网络深度" + str(encoder_depth), file=otfile)

    print("投影头维度" + str(embeddim_poj), file=otfile)
    print("投影头深度" + str(head_depth), file=otfile)
    
    print("dropout:" + str(dropout), file=otfile)
    print("feature_sparsity:" + str(feature_sparsity), file=otfile)
    print("learning rate:" + str(lr), file=otfile)
    print("temperature:" + str(temperature), file=otfile)
    print("hidden_layer_sizes:" + str(hidden_layer_sizes), file=otfile)
    print("max_iter:" + str(max_iter), file=otfile)

    the_labels = target['Label'].unique().tolist()
    if(the_labels[0] > the_labels[1]):
        the_labels[0], the_labels[1] = the_labels[1], the_labels[0]

    #标准化
    scaled_dataset = []
    scaler = StandardScaler()
    for dataset in Alldataset:
        scaled_data = scaler.fit_transform(dataset)
        scaled_df = pd.DataFrame(scaled_data, columns=dataset.columns, index=dataset.index)
        scaled_dataset.append(scaled_df)
    Alldataset = scaled_dataset

    #初始化评价指标
    Allaccuracy, Allsensitivity, Allspecific, AllAUC = {},{},{},{}
    for i, _ in enumerate(Alldataset):
        Allaccuracy[f'data{i}'], Allaccuracy[f'data{i}_emb'] = [],[]
        Allsensitivity[f'data{i}'], Allsensitivity[f'data{i}_emb'] = [],[]
        Allspecific[f'data{i}'], Allspecific[f'data{i}_emb'] = [],[]
        AllAUC[f'data{i}'], AllAUC[f'data{i}_emb'] = [],[]
    
    Allaccuracy['fusion'], Allaccuracy['fusion_emb'] = [],[]
    Allsensitivity['fusion'], Allsensitivity['fusion_emb'] = [],[]
    Allspecific['fusion'], Allspecific['fusion_emb'] = [],[]
    AllAUC['fusion'], AllAUC['fusion_emb'] = [],[]
    
    total_feature_importance = []
    count = 0

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=randomseed)
    for i in range(repeat):
        print("第"+ str(i) +"次五折交叉", file=otfile)
        print("第"+ str(i) +"次五折交叉")
        rng = np.random.default_rng()
        shuffled_indices = rng.permutation(len(target))
        Alldataset = [dataset.iloc[shuffled_indices].reset_index(drop=True) for dataset in Alldataset]
        target = target.iloc[shuffled_indices].reset_index(drop=True)
        ffcount = 0

        for count, (train_index, test_index) in enumerate(skf.split(Alldataset[0], target)):
            print("第"+ str(count) +"折")
            train_datas = [dataset.iloc[train_index] for dataset in Alldataset]
            test_datas = [dataset.iloc[test_index] for dataset in Alldataset]
            train_target, test_target = target.iloc[train_index], target.iloc[test_index]

            # to torch dataset
            train_ds = TabularDataset(
                        train_datas,
                        train_target.to_numpy()
                    )
            test_ds = TabularDataset(
                        test_datas,
                        test_target.to_numpy()
                    )

            print(f"The shape of data in train_ds are:\n{train_ds.shape}", file=otfile)
            print(f"The shape of data in test_ds are:\n{test_ds.shape}", file=otfile)


            #模型构建与参数设置
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            input_dims, emb_dims = [],[]
            for data in train_ds.data:
                input_dims.append(data.shape[1])
                #待定
                emb_dims.append(int(data.shape[1]/2))

            mm_model = MMSCARF(
                        input_dims = input_dims, 
                        emb_dims= emb_dims,
                        project_dim=embeddim_poj,
                        encoder_depth=encoder_depth,
                        head_depth=head_depth,
                        dropout=dropout,
                        dropout_poj=dropout_poj,
                        feature_sparsity=feature_sparsity,
                        learnable_sparsity=learnable_sparsity
                    ).to(device)

            optimizer = Adam(mm_model.parameters(), lr)
            mmcon_loss = SupConLoss(temperature, contrast_mode, base_temperature)
            loss_history = []

            #模型训练
            for epoch in range(1, epochs + 1):
                epoch_loss = train_epoch(mm_model, mmcon_loss, train_loader, optimizer, device, epoch)
                loss_history.append(epoch_loss)
            
            #获取表征结果与特征重要性排序
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

            train_embeddings = dataset_embeddings(mm_model, train_loader, device)
            test_embeddings = dataset_embeddings(mm_model, test_loader, device)

            train_datas = [data.values for data in train_datas]
            test_datas = [data.values for data in test_datas]

            total_feature_importance.append(mm_model.feature_importance())
            
            for i, array in enumerate(train_embeddings):
                print(f"Train_embedding {i + 1}:", end=" ", file=otfile)
                print("Shape:", array.shape, end=" ", file=otfile)
                print("Size:", array.size, file=otfile)
                
            for i, array in enumerate(test_embeddings):
                print(f"Test_embedding {i + 1}:", end=" ", file=otfile)
                print("Shape:", array.shape, end=" ", file=otfile)
                print("Size:", array.size, file=otfile)


            #fuse the two dataset
            train_datas_fusion = np.concatenate(train_datas, axis=1)
            test_datas_fusion = np.concatenate(test_datas, axis=1)

            train_embeddings_fusion = np.concatenate(train_embeddings, axis=1)
            test_embeddings_fusion = np.concatenate(test_embeddings, axis=1)

            print("原数据集:", file=otfile)
            print(train_datas_fusion.shape, file=otfile)
            print(test_datas_fusion.shape, file=otfile)
            print("训练后:", file=otfile)
            print(train_embeddings_fusion.shape, file=otfile)
            print(test_embeddings_fusion.shape, file=otfile)



            
            
            datasets = []

            for i in range(len(train_datas)):
                train_data = train_datas[i]
                test_data = test_datas[i]
                train_embedding = train_embeddings[i]
                test_embedding = test_embeddings[i]
                label = f'data{i}'
                
                dataset = (train_data, test_data, train_embedding, test_embedding, label)
                datasets.append(dataset)
            datasets.append((train_datas_fusion, test_datas_fusion, train_embeddings_fusion, test_embeddings_fusion, "fusion"))

            #svm_linear=svm.SVC(C=C_value, kernel=kenel_value,  decision_function_shape="ovo", probability=True)
            mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                    solver=solver, alpha=alpha, max_iter=max_iter, random_state=randomseed)
            for i, dataset in enumerate(datasets):
                train_data, test_data, train_embeddings, test_embeddings, dataset_name = dataset         
                
                # Train the classifier on the original data
                mlp.fit(train_data, train_target.values.ravel())
                vanilla_predictions = mlp.predict(test_data)
                probs = mlp.predict_proba(test_data)[:, 1]
                print("------基于{}------".format(dataset_name), file=otfile)
                acc, sen, spe, auc = evaluate_model(test_target, vanilla_predictions, probs, the_labels, otfile)
 
                Allaccuracy[dataset_name].append(acc)
                Allsensitivity[dataset_name].append(sen)
                Allspecific[dataset_name].append(spe)
                AllAUC[dataset_name].append(auc)

                mlp.fit(train_embeddings, train_target.values.ravel())
                vanilla_predictions = mlp.predict(test_embeddings)
                probs = mlp.predict_proba(test_embeddings)[:, 1]
                print("------基于{}_emb------".format(dataset_name), file=otfile)
                if len(np.unique(test_target)) == 2:
                    vanilla_predictions = change_threshold(mlp, test_embeddings, test_target, 0.1, the_labels, otfile)
                acc, sen, spe, auc = evaluate_model(test_target, vanilla_predictions, probs, the_labels, otfile)
                print("\n")
                dataset_name = dataset_name + "_emb"
                Allaccuracy[dataset_name].append(acc)
                Allsensitivity[dataset_name].append(sen)
                Allspecific[dataset_name].append(spe)
                AllAUC[dataset_name].append(auc)
                
            
            count = count + 1
            ffcount = ffcount + 1
            

        
    for i in range(len(Allaccuracy)//2):
        if ((i+1)*2 == len(Allaccuracy)):
            org_name = f'fusion'
            emb_name = f'fusion_emb'
        else:
            org_name = f'data{i}'
            emb_name = f'data{i}_emb'
        
        acc_org = np.array(Allaccuracy[org_name])
        acc_emb = np.array(Allaccuracy[emb_name])
        sen_org = np.array(Allsensitivity[org_name])
        sen_emb = np.array(Allsensitivity[emb_name])
        spe_org = np.array(Allspecific[org_name])
        spe_emb = np.array(Allspecific[emb_name])
        auc_org = np.array(AllAUC[org_name])
        auc_emb = np.array(AllAUC[emb_name])
        
        acc_org_mean = np.mean(acc_org)
        acc_org_std = np.std(acc_org)
        acc_emb_mean = np.mean(acc_emb)
        acc_emb_std = np.std(acc_emb)
        
        sen_org_mean = np.mean(sen_org)
        sen_org_std = np.std(sen_org)
        sen_emb_mean = np.mean(sen_emb)
        sen_emb_std = np.std(sen_emb)
        
        spe_org_mean = np.mean(spe_org)
        spe_org_std = np.std(spe_org)
        spe_emb_mean = np.mean(spe_emb)
        spe_emb_std = np.std(spe_emb)
        
        auc_org_mean = np.mean(auc_org)
        auc_org_std = np.std(auc_org)
        auc_emb_mean = np.mean(auc_emb)
        auc_emb_std = np.std(auc_emb)
        
        print(f"————{org_name}————", file=otfile)
        print(f"Accuracy - 平均值: {acc_org_mean}, 标准差: {acc_org_std}", file=otfile)
        print(f"Sensitivity - 平均值: {sen_org_mean}, 标准差: {sen_org_std}", file=otfile)
        print(f"Specificity - 平均值: {spe_org_mean}, 标准差: {spe_org_std}", file=otfile)
        print(f"AUC - 平均值: {auc_org_mean}, 标准差: {auc_org_std}", file=otfile)
        print("\n", file=otfile)
        print(f"————{emb_name}————", file=otfile)
        print(f"Accuracy - 平均值: {acc_emb_mean}, 标准差: {acc_emb_std}", file=otfile)
        print(f"Sensitivity - 平均值: {sen_emb_mean}, 标准差: {sen_emb_std}", file=otfile)
        print(f"Specificity - 平均值: {spe_emb_mean}, 标准差: {spe_emb_std}", file=otfile)
        print(f"AUC - 平均值: {auc_emb_mean}, 标准差: {auc_emb_std}", file=otfile)
        print("\n", file=otfile)
    
    #保存特征重要性
    num_trainings = len(total_feature_importance)
    num_datasets = len(total_feature_importance[0])
    avg_feature_importance = []
    for dataset_index in range(num_datasets):
        dataset_trainings = [total_feature_importance[i][dataset_index] for i in range(num_trainings)]
        avg_importance = np.mean(dataset_trainings, axis=0)
        avg_feature_importance.append(avg_importance)

    avg_feature_importance_df = pd.DataFrame(avg_feature_importance)
    avg_feature_importance_df.to_csv(importanceFilepath, index=False)

    otfile.close