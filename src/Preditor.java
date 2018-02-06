import java.io.File;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.TreeMap;
import java.util.TreeSet;
import weka.classifiers.AbstractClassifier;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

//https://weka.wikispaces.com/Use+Weka+in+your+Java+code
public class Preditor {

	static TreeSet<String> stopwords;
	private static Instances instances;
	static AbstractClassifier[] classifiers;
	static TraineUtils traineUtils = TraineUtils.newInstance();
	
	public static void main(String[] args) {
		predictAndReplaceFiles();
	}
	
	public static void predictAndReplaceFiles() {
		try {
			stopwords = traineUtils.loadStopwords();
			String[] inputs = traineUtils.getInputs().list();
			File classifiedsRootFolder = traineUtils.getClassifieds();
			
			instances = traineUtils.loadWekaDataSource();
			classifiers = traineUtils.loadSavedModels();
			
			for(String input: inputs) {
				
				String toPredictPath = traineUtils.getInputs().getAbsolutePath() + File.separator + input;
				
				System.out.println("============= "+input + " ===============");
				
				int[] classVotes = new int[instances.classAttribute().numValues()]; 
				
				for(AbstractClassifier classifier: classifiers) {
					int vote = predict(toPredictPath, classifier);	
					classVotes[vote] = classVotes[vote] + 1;
				}
				int moreVotedPosition = -1;
				int moreVotes = -1;
				int count = 0;
				for(int value : classVotes) {
					if(value > moreVotes) {
						moreVotes = value;
						moreVotedPosition = count;
					}
					count++;
				}
				
				System.out.println("Voted class : "+ instances.classAttribute().value(moreVotedPosition));
				
				File classiFiedsFolder = new File(classifiedsRootFolder, instances.classAttribute().value(moreVotedPosition));
				classiFiedsFolder.mkdirs();
				
				File predictedFile = new File(toPredictPath);
				File replacedFile = new File(classiFiedsFolder, input);
				if(!replacedFile.exists()) {
					traineUtils.moveFile(predictedFile, replacedFile);
					predictedFile.deleteOnExit();
				}
			}	
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static int predict(String entrada, AbstractClassifier classifier) {
		int betterClass = -1; 
		try {
			double maxValue = 0.0d;
			
			File inputFile = new File(entrada);
			if(!inputFile.exists()) throw new Exception("File no exists: "+inputFile.getAbsolutePath());
			String txt = traineUtils.extractFromPdf(inputFile,stopwords);
			TreeMap<String, Integer> itemStatistics = new TreeMap<>();
			traineUtils.extractWordsStatistcs(itemStatistics, txt);
			
			Instance newEntitiy = new DenseInstance(instances.numAttributes());			
			newEntitiy.setDataset(instances);            
            for(int i = 0; i < instances.numAttributes() - 1; i++) {
            	Integer qtd = itemStatistics.get(instances.attribute(i).name());
            	newEntitiy.setValue(i, qtd!= null? qtd :0 );
            }
            
            double resultado[] = classifier.distributionForInstance(newEntitiy);
            DecimalFormat format = (DecimalFormat)DecimalFormat.getInstance();
            format.setMaximumFractionDigits(4);
            
            int classId = 0;
            for(double result : resultado) {
            	if(maxValue < result) {
            		maxValue = result;
            		betterClass = classId;
            	}
            	System.out.println(classifier.getClass().getName()+" - "+instances.classAttribute().value(classId)+" "+format.format(result));
            	classId++;
            }
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return betterClass;
	}
}