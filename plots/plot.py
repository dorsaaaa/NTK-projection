import matplotlib.pyplot as plt
import pandas as pd

samples = [100,200,300,400,500,600]



single_ID = pd.read_csv('single_task_results_diff_sample.csv',index_col=None)
meta_ELLA = pd.read_csv('MetaELLA_results_diff_samples.csv',index_col=None)
single_ID_300 = single_ID.iloc[:18]
single_ID_600 =single_ID.iloc[18:]

#SINGLE_300
'''
plt.figure(figsize=(10, 6))
plt.plot(samples, list(single_ID['train_acc']), label='Train Accuracy', marker='o', color='b')
plt.plot(samples, list(single_ID['test_acc']), label='Test Accuracy', marker='o', color='r')
plt.xlabel('Number of Samples')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy vs. Number of Samples for single Task ID')
plt.legend()
plt.grid(True)
'''
grouped = single_ID_300.groupby('num_samples').agg(
    train_mean=('train_acc', 'mean'),
    train_std=('train_acc', 'std'),
    test_mean=('test_acc', 'mean'),
    test_std=('test_acc', 'std')
).reset_index()


plt.figure(figsize=(10, 6))
plt.errorbar(
    grouped['num_samples'], grouped['train_mean'], yerr=grouped['train_std'], 
    fmt='-o', capsize=5, label='Train Accuracy', color='blue'
)
plt.errorbar(
    grouped['num_samples'], grouped['test_mean'], yerr=grouped['test_std'], 
    fmt='-o', capsize=5, label='Test Accuracy', color='green'
)

plt.xlabel('Number of Samples')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy for single task ID 300')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_vs_samples_ID_single_300.pdf', format='pdf')


#SINGLE_600
grouped = single_ID_600.groupby('num_samples').agg(
    train_mean=('train_acc', 'mean'),
    train_std=('train_acc', 'std'),
    test_mean=('test_acc', 'mean'),
    test_std=('test_acc', 'std')
).reset_index()

plt.figure(figsize=(10, 6))
plt.errorbar(
    grouped['num_samples'], grouped['train_mean'], yerr=grouped['train_std'], 
    fmt='-o', capsize=5, label='Train Accuracy', color='blue'
)
plt.errorbar(
    grouped['num_samples'], grouped['test_mean'], yerr=grouped['test_std'], 
    fmt='-o', capsize=5, label='Test Accuracy', color='green'
)

plt.xlabel('Number of Samples')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy for single task ID 600')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_vs_samples_ID_single_600.pdf', format='pdf')


#META
grouped = meta_ELLA.groupby('num_samples').agg(
    training_train_mean=('avg training train_acc', 'mean'),
    training_train_std=('avg training train_acc', 'std'),

    training_test_mean=('avg training test_acc', 'mean'),
    training_test_std=('avg training test_acc', 'std'),

    testing_train_mean=('avg testing train_acc', 'mean'),
    testing_train_std=('avg testing train_acc', 'std'),

    testing_test_mean=('avg testing test_acc', 'mean'),
    testing_test_std=('avg testing test_acc', 'std'),
).reset_index()

plt.figure(figsize=(10, 6))
plt.errorbar(
    grouped['num_samples'], grouped['training_train_mean'], yerr=grouped['training_train_std'], 
    fmt='-o', capsize=5, label='Training Train Accuracy', color='blue'
)
plt.errorbar(
    grouped['num_samples'], grouped['training_test_mean'], yerr=grouped['training_test_std'], 
    fmt='-o', capsize=5, label='Training Test Accuracy', color='green'
)
plt.errorbar(
    grouped['num_samples'], grouped['testing_train_mean'], yerr=grouped['testing_train_std'], 
    fmt='-o', capsize=5, label='Testing Train Accuracy', color='red'
)
plt.errorbar(
    grouped['num_samples'], grouped['testing_test_mean'], yerr=grouped['testing_test_std'], 
    fmt='-o', capsize=5, label='Testing Test Accuracy', color='purple'
)

plt.xlabel('Number of Samples')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy for ELLA')
plt.legend()
plt.grid(True)

plt.savefig('accuracy_vs_samples_ELLA_Meta.pdf', format='pdf')


