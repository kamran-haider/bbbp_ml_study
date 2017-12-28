Dear Candidate!

This problem is designed to test your skill in building predictive models for chemical properties.
You are presented with a real world dataset of chemicals, which have been labelled as being blood brain barrier penetration positive (1) or negative (0).
Your task is to build a model that can predict BBB penetration or not, and have that model be able to generalise to new unseen molecules effectively.
The dataset is presented in the following format:

SMILES	Label
[S](=O)(=O)(OCCCCO[S](=O)(=O)C)C	0
C[C@@H]4C[C@H]3[C@@H]2C[C@H](F)C1=CC(=O)C=C[C@]1(C)[C@@]2(F)[C@@H](O)C[C@]3(C)[C@@]4(OC(=O)C5=CC=CO5)C(=O)SCF	0
C1(=NCCO1)NC(C2CC2)C3CC3	0
C24=C(SC1=CC=CC=C1N2CCCN3CCC(CCO)CC3)C=CC(=C4)[S](N(C)C)(=O)=O	1
C2CC1=CC(=CC(=C1C(C3=CC=CC=C23)[N]4C=CN=C4)Cl)Cl	0
O=C(N1C[C@@H]2[C@H](C1)CCCC2)C[C@@H](C(O)=O)CC3=CC=CC=C3	0
[C@@]12([C@H]([C@H](N(C)C)C(=C(C1=O)C(=O)N)O)[C@@H](O)[C@H]3C(=C2O)C(C4=C([C@@H]3C)C=CC=C4O)=O)O	0
With the chemical SMILES string representation and a label of '1' for BBB positive and '0' for BBB negative. How you chose to convert or process the chemical structures is up to you

You have been given two files, bbb_train.csv which is the training dataset and bbb_test.csv which is a test set with the labels removed. You need to submit your predictions for the test file in the same format as the train.
Your task is as follows:
Build and validate a predictive model on this dataset
You can use any method and representation of the data you feel is appropriate
Prepare a presentation to discuss your method and results (and any other methods you feel could work well here), please include:
Slides on how your chemical representation worked
Slides on how the model works
Slides on results and model validation, how the validation works, what metrics are useful here, what to lookout for when validating chemical models
Slides on how you could improve this method
Package your code, your trained models, your prediction file, and presentation into a zip file and send it to us.

The prediction file should be a csv file with the same structure as the training dataset

Good Luck!
