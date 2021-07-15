using System;
using System.Collections;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Windows.Forms;
using weka.core;

namespace ML_CL
{
    public partial class Form1 : Form
    {
        private static Tuple<weka.classifiers.Classifier, double> result;
        private string path;
        private string output_filename;       

  
        private const int percentSplit = 66;
        static weka.core.Instances insts = null;

        static weka.classifiers.Classifier cl_BEST = null;

        static weka.classifiers.Classifier cl_NB = null;
        static weka.classifiers.Classifier cl_LOG = null;
        static weka.classifiers.Classifier cl_1NN = null;
        static weka.classifiers.Classifier cl_3NN = null;
        static weka.classifiers.Classifier cl_5NN = null;
        static weka.classifiers.Classifier cl_7NN = null;
        static weka.classifiers.Classifier cl_9NN = null;
        static weka.classifiers.Classifier cl_RF = null;
        static weka.classifiers.Classifier cl_RT = null;
        static weka.classifiers.Classifier cl_J48 = null;
        static weka.classifiers.Classifier cl_MLP = null;
        static weka.classifiers.Classifier cl_SVM = null;


        public Form1()
        {
            InitializeComponent();
        }
        private void Form1_Load(object sender, EventArgs e)
        {
        }
        private void Button1_Click(object sender, EventArgs e)
        {
            OpenFileDialog file = new OpenFileDialog
            {
                InitialDirectory = Directory.GetParent(System.Environment.CurrentDirectory).Parent.Parent.FullName,
                Filter = "Arff Dosyası |*.arff"
            };


            if (file.ShowDialog() == DialogResult.OK)
            {
                path = file.FileName;
                string filename = Path.GetFileName(path);
                output_filename = path.Substring(0, path.Length - 5) + "_temp" + path.Substring(path.Length - 5, 5);
                string ext = filename.Split('.').Last();

                if (ext.Equals("arff"))
                {
                    label2.Visible = false;
                    button2.Visible = false;
                    GridView.Columns.Clear();
                    GridView.Refresh();

                    textBox1.Text = filename;
                    insts = new weka.core.Instances(new java.io.FileReader(path));
                    result = FindMostSuccesfull();
                    Grid(result);
                }

                else
                {
                    MessageBox.Show("The format must be .arff");
                }
            }
        }
        private void Button2_Click(object sender, EventArgs e)
        {
            Instances insts = new Instances(new java.io.FileReader(path));
            PredFile(insts);
            Instance newInstance = new DenseInstance(insts.numAttributes() - 1);
            insts.setClassIndex(insts.numAttributes() - 1);

            bool error_occured = false;
            for (int i = 0; i < GridView.Rows[0].Cells.Count - 1; i++)
            {
                try
                {

                    if (GridView.Rows[0].Cells[i].Value != null)
                        if (insts.attribute(i).isNumeric())
                            newInstance.setValue(insts.attribute(i), Convert.ToDouble(GridView.Rows[0].Cells[i].Value.ToString()));
                        else
                            newInstance.setValue(insts.attribute(i), GridView.Rows[0].Cells[i].Value.ToString());

                    else
                    {
                        MessageBox.Show("You need to fill all columns");

                        error_occured = true;
                        break;
                    }
                }
                catch (FormatException)
                {
                    MessageBox.Show("Please fill columns with correct value !!");
                    break;
                }

            }
            if (error_occured)
                return;

            Instances temp_insts = new Instances(new java.io.FileReader(output_filename));
            temp_insts.setClassIndex(temp_insts.numAttributes() - 1);
            newInstance.setDataset(temp_insts);
            InsertPrediction(newInstance);
            temp_insts = new Instances(new java.io.FileReader(output_filename));
            temp_insts.setClassIndex(temp_insts.numAttributes() - 1);


            double predictedClass = 0;
            try
            {
                predictedClass = cl_BEST.classifyInstance(temp_insts.instance(0));
            }
            catch (IndexOutOfRangeException ex)
            {
                Console.WriteLine(ex.StackTrace);

            }
            insts = new Instances(new java.io.FileReader(path));
            label2.Text = "RESULT : " + insts.attribute(insts.numAttributes() - 1).value(Convert.ToInt32(predictedClass)).ToString();
            label2.Visible = true;

            GridView.Rows[0].Cells[GridView.ColumnCount - 1].Value = insts.attribute(insts.numAttributes() - 1).value(Convert.ToInt32(predictedClass)).ToString();
            GridView.Rows[0].Cells[GridView.ColumnCount - 1].ReadOnly = true;
            GridView.Columns[GridView.ColumnCount - 1].Visible = true;


        }
        private void Grid(Tuple<weka.classifiers.Classifier, double> result)
        {
            Instances insts = new Instances(new java.io.FileReader(path));

            cl_BEST = result.Item1;
            string cl_name;
            try                                            
            {
                var WordsArray = cl_BEST.ToString().Split();
                cl_name = WordsArray[0] + ' ' + WordsArray[1] + ' ' + WordsArray[2];
            }
            catch (NullReferenceException)
            {
                MessageBox.Show("Data set does not have enough columns.");
                return;
            }

            label1.Text = cl_name + " is the most succesfull algorithm for this data set(% " + String.Format("{0:0.00}", result.Item2) + " )";
            label1.Visible = true;


            DataGridViewComboBoxColumn combo = new DataGridViewComboBoxColumn();
            var column = new DataGridViewTextBoxColumn();
 
            ArrayList row = new ArrayList();
            GridView.RowCount = 1;


            for (int i = 0; i < insts.numAttributes() - 1; i++)
            {
                if (insts.attribute(i).isNominal())
                {
                    combo = new DataGridViewComboBoxColumn
                    {
                        HeaderText = insts.attribute(i).name()
                    };

                    for (int j = 0; j < insts.attribute(i).numValues(); j++)
                        row.Add(insts.attribute(i).value(j).ToString());

                    combo.Items.AddRange(row.ToArray());
                    GridView.Columns.Add(combo);
                }

                else
                {
                    column = new DataGridViewTextBoxColumn
                    {
                        HeaderText = insts.attribute(i).name(),
                        Name = insts.attribute(i).name()
                    };

                    GridView.Columns.Add(column);
                }

                row.Clear();
            }

            column = new DataGridViewTextBoxColumn
            {
                HeaderText = insts.attribute(insts.numAttributes() - 1).name(),
                Name = insts.attribute(insts.numAttributes() - 1).name(),
                Visible = false
            };
            GridView.Columns.Add(column);

            GridView.Visible = true;
            button2.Visible = true;
            GridView.Columns.RemoveAt(0);

        }
        private void InsertPrediction(Instance Instance)
        {
            string data = "";
            string specifier = "G";
            for (int i = 0; i < Instance.numAttributes(); i++)
            {
                if (Instance.attribute(i).isNumeric())
                    data += Convert.ToDouble(Instance.value(i)).ToString(specifier, CultureInfo.InvariantCulture) + " ,";
                else
                    data += Instance.attribute(i).value(Convert.ToInt32(Instance.value(i))).ToString() + " ,";
            }

            data += "?\n";
            try
            {
                using (StreamWriter sw = File.AppendText(output_filename))
                    sw.Write(data);

            }
            catch (Exception Ex)
            {
                Console.WriteLine(Ex.ToString());
            }
        }
        private void PredFile(Instances insts)
        {
            string data = "";
            string line = "";
            try
            {
                using (StreamReader sr = File.OpenText(path))
                    while ((line = sr.ReadLine()) != null & line.ToUpper() != "@data".ToUpper())
                        data += line + "\n";

                data += "@DATA\n";

                using (StreamWriter sw = File.CreateText(output_filename))
                    sw.Write(data);

            }
            catch (Exception Ex)
            {
                Console.WriteLine(Ex.ToString());
            }
        }

        public static double ClassTest_1NN(weka.core.Instances insts)
        {
            try
            {
                insts.setClassIndex(insts.numAttributes() - 1);

                cl_1NN = new weka.classifiers.lazy.IBk(1);

                weka.filters.Filter myNormalize = new weka.filters.unsupervised.attribute.Normalize();
                myNormalize.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myNormalize);

                weka.filters.Filter myDummyAttr = new weka.filters.unsupervised.attribute.NominalToBinary();
                myDummyAttr.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myDummyAttr);

                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

                cl_1NN.buildClassifier(train);


                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                    double predictedClass = cl_1NN.classifyInstance(currentInst);
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                return (double)numCorrect / (double)testSize * 100.0;
            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
                return 0;
            }
        }

        public static double ClassTest_3NN(weka.core.Instances insts)
        {
            try
            {
                insts.setClassIndex(insts.numAttributes() - 1);

                cl_3NN = new weka.classifiers.lazy.IBk(3);

                weka.filters.Filter myNormalize = new weka.filters.unsupervised.attribute.Normalize();
                myNormalize.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myNormalize);

                weka.filters.Filter myDummyAttr = new weka.filters.unsupervised.attribute.NominalToBinary();
                myDummyAttr.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myDummyAttr);

                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

                cl_3NN.buildClassifier(train);


                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                    double predictedClass = cl_3NN.classifyInstance(currentInst);
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                return (double)numCorrect / (double)testSize * 100.0;
            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
                return 0;
            }
        }

        public static double ClassTest_5NN(weka.core.Instances insts)
        {
            try
            {
                insts.setClassIndex(insts.numAttributes() - 1);

                cl_5NN = new weka.classifiers.lazy.IBk(5);

                weka.filters.Filter myNormalize = new weka.filters.unsupervised.attribute.Normalize();
                myNormalize.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myNormalize);

                weka.filters.Filter myDummyAttr = new weka.filters.unsupervised.attribute.NominalToBinary();
                myDummyAttr.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myDummyAttr);

                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

                cl_5NN.buildClassifier(train);


                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                    double predictedClass = cl_5NN.classifyInstance(currentInst);
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                return (double)numCorrect / (double)testSize * 100.0;
            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
                return 0;
            }
        }

        public static double ClassTest_7NN(weka.core.Instances insts)
        {
            try
            {
                insts.setClassIndex(insts.numAttributes() - 1);

                cl_7NN = new weka.classifiers.lazy.IBk(7);

                weka.filters.Filter myNormalize = new weka.filters.unsupervised.attribute.Normalize();
                myNormalize.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myNormalize);

                weka.filters.Filter myDummyAttr = new weka.filters.unsupervised.attribute.NominalToBinary();
                myDummyAttr.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myDummyAttr);

                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

                cl_7NN.buildClassifier(train);


                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                    double predictedClass = cl_7NN.classifyInstance(currentInst);
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                return (double)numCorrect / (double)testSize * 100.0;
            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
                return 0;
            }
        }

        public static double ClassTest_9NN(weka.core.Instances insts)
        {
            try
            {
                insts.setClassIndex(insts.numAttributes() - 1);

                cl_9NN = new weka.classifiers.lazy.IBk(9);

                weka.filters.Filter myNormalize = new weka.filters.unsupervised.attribute.Normalize();
                myNormalize.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myNormalize);

                weka.filters.Filter myDummyAttr = new weka.filters.unsupervised.attribute.NominalToBinary();
                myDummyAttr.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myDummyAttr);

                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

                cl_9NN.buildClassifier(train);


                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                    double predictedClass = cl_9NN.classifyInstance(currentInst);
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                return (double)numCorrect / (double)testSize * 100.0;
            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
                return 0;
            }
        }

        public static double ClassTest_NB(weka.core.Instances insts)
        {
            try
            {
                insts.setClassIndex(insts.numAttributes() - 1);

                cl_NB = new weka.classifiers.bayes.NaiveBayes();

                weka.filters.Filter myDiscretize = new weka.filters.unsupervised.attribute.Discretize();
                myDiscretize.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myDiscretize);

                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

                cl_NB.buildClassifier(train);


                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                    double predictedClass = cl_NB.classifyInstance(currentInst);
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                return (double)numCorrect / (double)testSize * 100.0;
            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
                return 0;
            }
        }

        public static double ClassTest_LOG(weka.core.Instances insts)
        {
            try
            {
                insts.setClassIndex(insts.numAttributes() - 1);

                cl_LOG = new weka.classifiers.functions.Logistic();

                weka.filters.Filter myNormalize = new weka.filters.unsupervised.attribute.Normalize();
                myNormalize.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myNormalize);

                weka.filters.Filter myDummyAttr = new weka.filters.unsupervised.attribute.NominalToBinary();
                myDummyAttr.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myDummyAttr);

                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

                cl_LOG.buildClassifier(train);


                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                    double predictedClass = cl_LOG.classifyInstance(currentInst);
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                return (double)numCorrect / (double)testSize * 100.0;
            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
                return 0;
            }
        }

        public static double ClassTest_J48(weka.core.Instances insts)
        {
            try
            {
                insts.setClassIndex(insts.numAttributes() - 1);

                cl_J48 = new weka.classifiers.trees.J48();

                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

                cl_J48.buildClassifier(train);


                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                    double predictedClass = cl_J48.classifyInstance(currentInst);
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                return (double)numCorrect / (double)testSize * 100.0;
            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
                return 0;
            }
        }

        public static double ClassTest_RF(weka.core.Instances insts)
        {
            try
            {
                insts.setClassIndex(insts.numAttributes() - 1);

                cl_RF = new weka.classifiers.trees.RandomForest();

                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

                cl_RF.buildClassifier(train);


                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                    double predictedClass = cl_RF.classifyInstance(currentInst);
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                return (double)numCorrect / (double)testSize * 100.0;
            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
                return 0;
            }
        }

        public static double ClassTest_RT(weka.core.Instances insts)
        {
            try
            {
                insts.setClassIndex(insts.numAttributes() - 1);

                cl_RT = new weka.classifiers.trees.RandomTree();

                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

                cl_RT.buildClassifier(train);


                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                    double predictedClass = cl_RT.classifyInstance(currentInst);
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                return (double)numCorrect / (double)testSize * 100.0;
            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
                return 0;
            }
        }

        public static double ClassTest_MLP(weka.core.Instances insts)
        {
            try
            {
                insts.setClassIndex(insts.numAttributes() - 1);

                cl_MLP = new weka.classifiers.functions.MultilayerPerceptron();

                weka.filters.Filter myNormalize = new weka.filters.unsupervised.attribute.Normalize();
                myNormalize.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myNormalize);

                weka.filters.Filter myDummyAttr = new weka.filters.unsupervised.attribute.NominalToBinary();
                myDummyAttr.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myDummyAttr);

                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

                cl_MLP.buildClassifier(train);


                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                    double predictedClass = cl_MLP.classifyInstance(currentInst);
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                return (double)numCorrect / (double)testSize * 100.0;
            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
                return 0;
            }
        }

        public static double ClassTest_SVM(weka.core.Instances insts)
        {
            try
            {
                insts.setClassIndex(insts.numAttributes() - 1);

                cl_SVM = new weka.classifiers.functions.SMO();

                weka.filters.Filter myNormalize = new weka.filters.unsupervised.attribute.Normalize();
                myNormalize.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myNormalize);

                weka.filters.Filter myDummyAttr = new weka.filters.unsupervised.attribute.NominalToBinary();
                myDummyAttr.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myDummyAttr);

                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

                cl_SVM.buildClassifier(train);


                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                    double predictedClass = cl_SVM.classifyInstance(currentInst);
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                return (double)numCorrect / (double)testSize * 100.0;
            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
                return 0;
            }
        }
        public Tuple<weka.classifiers.Classifier, double> FindMostSuccesfull()
        {

            weka.core.Instances insts = new weka.core.Instances(new java.io.FileReader(path));
            weka.core.Instances insts_temp = new weka.core.Instances(new java.io.FileReader(path));

            double max_acc = ClassTest_1NN(insts);
            cl_BEST = cl_1NN;

            insts = insts_temp;
            double temp_acc = ClassTest_3NN(insts);

            if (temp_acc > max_acc)
            {
                max_acc = temp_acc;
                cl_BEST = cl_3NN;
            }

            insts = insts_temp;
            temp_acc = ClassTest_5NN(insts);

            if (temp_acc > max_acc)
            {
                max_acc = temp_acc;
                cl_BEST = cl_5NN;
            }

            insts = insts_temp;
            temp_acc = ClassTest_7NN(insts);

            if (temp_acc > max_acc)
            {
                max_acc = temp_acc;
                cl_BEST = cl_7NN;
            }

            insts = insts_temp;
            temp_acc = ClassTest_9NN(insts);

            if (temp_acc > max_acc)
            {
                max_acc = temp_acc;
                cl_BEST = cl_9NN;
            }

            insts = insts_temp;
            temp_acc = ClassTest_NB(insts);

            if (temp_acc > max_acc)
            {
                max_acc = temp_acc;
                cl_BEST = cl_NB;
            }

            insts = insts_temp;
            temp_acc = ClassTest_J48(insts);

            if (temp_acc > max_acc)
            {
                max_acc = temp_acc;
                cl_BEST = cl_J48;
            }

            insts = insts_temp;
            temp_acc = ClassTest_RT(insts);

            if (temp_acc > max_acc)
            {
                max_acc = temp_acc;
                cl_BEST = cl_RT;
            }

            insts = insts_temp;
            temp_acc = ClassTest_RF(insts);

            if (temp_acc > max_acc)
            {
                max_acc = temp_acc;
                cl_BEST = cl_RF;
            }

            insts = insts_temp;
            temp_acc = ClassTest_MLP(insts);

            if (temp_acc > max_acc)
            {
                max_acc = temp_acc;
                cl_BEST = cl_MLP;
            }

            insts = insts_temp;
            temp_acc = ClassTest_SVM(insts);

            if (temp_acc > max_acc)
            {
                max_acc = temp_acc;
                cl_BEST = cl_SVM;
            }

            insts = insts_temp;
            temp_acc = ClassTest_LOG(insts);

            if (temp_acc > max_acc)
            {
                max_acc = temp_acc;
                cl_BEST = cl_LOG;
            }

            return new Tuple<weka.classifiers.Classifier, double>(cl_BEST, max_acc);

        }

    }

}


