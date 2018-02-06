import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.TreeSet;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.rules.JRip;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class TraineUtils {
	private static File outputDir = new File("res/generated");
	private static File dataDir = new File("res/traineset");
	private static File stopwordsFile = new File("res/stopwords.txt");
	private static File trainedFile = new File(outputDir,"treined.arff");
	private static File inputs = new File("res/toclassify");
	private static File classifieds = new File("res/classifieds");
	
	static AbstractClassifier[] classifiers = new AbstractClassifier[]{new NaiveBayes(), new NaiveBayesMultinomial(), new JRip()}; 
	static String[] classifiersFileName = new String[]{"naive_bayes.model","mn_naive_bayes.model", "jrip.model"}; 
	
	public static TraineUtils newInstance() {
		return new TraineUtils();
	}
	
	private TraineUtils() {
		if(!outputDir.exists()) outputDir.mkdirs();
		if(!dataDir.exists()) dataDir.mkdirs();
		if(!inputs.exists()) inputs.mkdirs();
		if(!classifieds.exists()) inputs.mkdirs();
	}
	
	public File getOutputDir() {
		return outputDir;
	}
	
	public File getDataDir() {
		return dataDir;
	}
	
	public File getInputs() {
		return inputs;
	}
	
	public File getClassifieds() {
		return classifieds;
	}
	
	public File getStopwordsFile() {
		return stopwordsFile;
	}
	
	public File getTrainedFile() {
		return trainedFile;
	}
	
	public AbstractClassifier[] loadSavedModels() {
		
		AbstractClassifier[] loadedClassfiers = new AbstractClassifier[classifiersFileName.length];
		
		for(int i = 0; i < classifiersFileName.length; i++) {
			ObjectInputStream model;
			try {
				model = new ObjectInputStream(new FileInputStream(new File(outputDir, classifiersFileName[i])));
				loadedClassfiers[i] = (AbstractClassifier)model.readObject();
				model.close();
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		return loadedClassfiers;
	}

	public TreeSet<String> loadStopwords() {
		TreeSet<String> stopwords = new TreeSet<String>();
		String txt = readFromFile(stopwordsFile);
		String[] tokens = txt.split("\n");
		stopwords.addAll(Arrays.asList(tokens));
		stopwords.addAll(Arrays.asList(new String[] {""," ","-","(+)","(-)","(*)","(/)"}));
		return stopwords;
	}
	
	public void generateModels() {
		try {
			Instances instances = loadWekaDataSource();			
	        for(int i = 0; i < classifiersFileName.length; i++) {
	        	AbstractClassifier classifier =  classifiers[i];
				classifier.buildClassifier(instances);
				
		        ObjectOutputStream output = new ObjectOutputStream(new FileOutputStream(new File(outputDir,classifiersFileName[i])));
		        output.writeObject(classifier);
		        output.flush();
		        output.close();	        	
	        }
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public void generateArff(File classesDir, File outputDir, TreeMap<String, Integer> generalStatistics) {
		StringBuffer classes = new StringBuffer();
		for(File f : classesDir.listFiles()) {
			if(!f.isDirectory())continue;
			if(classes.length()>1){
				classes.append(",");				
			}
			classes.append(f.getName());
		}
		
		String fileName = trainedFile.getName();
		saveTxtToFile(outputDir, fileName, "@relation comprovante-conta\n");
		
		List<String> sortedGeneralLabels = sortStatistics(generalStatistics);
		StringBuffer strB = new StringBuffer();
		for(String key: sortedGeneralLabels) {
			strB.append("@attribute '"+key.trim().concat("' NUMERIC\n"));
		}
		strB.append("@attribute class {"+classes.toString()+"}\n");
		
		saveTxtToFile(outputDir,fileName, strB.toString());
		saveTxtToFile(outputDir,fileName, "@data\n");
		
		for(File f: outputDir.listFiles()) {
			if(f.isDirectory() || !f.getName().endsWith(".txt")) continue;
			String txt = readFromFile(f);
			TreeMap<String, Integer> itemStatistics = new TreeMap<>();
			extractWordsStatistcs(itemStatistics, txt);
			System.out.println("==============================");
			List<String> itemSortedLabels =  sortStatistics(itemStatistics);
			String strItemDescriptorsValues = generateItemMap(sortedGeneralLabels, itemSortedLabels, itemStatistics);
			String classe = f.getName().split("_")[0];
			saveTxtToFile(outputDir,fileName, strItemDescriptorsValues.concat(classe).concat("\n"));			
		}
	}
	
	public String generateItemMap(List<String> generalLabels, List<String> itemLabels, TreeMap<String, Integer> itemStatistics) {
		StringBuffer strLabelValues = new StringBuffer();
		for(String key: generalLabels) {			
			if(itemLabels.contains(key)) {
				strLabelValues.append(itemStatistics.get(key));				
			}else{
				strLabelValues.append(0);
			}
			strLabelValues.append(",");
		}
		return strLabelValues.toString();
	}
	
	public List<String> sortStatistics(TreeMap<String, Integer> wordStatistics) {
		List<String> sortedValues = new ArrayList<>();
		sortedValues.addAll(wordStatistics.keySet());
		
		Collections.sort(sortedValues, new Comparator<String>() {
			@Override
			public int compare(String o1, String o2) {
				return wordStatistics.get(o2).compareTo(wordStatistics.get(o1));
			}
		});
		return sortedValues;
	}
	
	public void clearDir(File dir) {
		for(File f : dir.listFiles()) {
			f.delete();
		}
	}
	
	public String removeStopwords(String str, TreeSet<String> stopwords) {
		String[] tokens = str.split(" ");
		StringBuffer strB = new StringBuffer();
		for(String token: tokens) {
			if(!stopwords.contains(token)) strB.append(token+" ");
		}	
		return strB.toString();
	}
	
	String removeTrash(String str) {
		String[] charTOremove = new String[] {
				"(",")","/",":",".","'",";","-","_","%",
				"$","#","@","!","&","*","+","=","?","­ "," ",
				"1","2","3","4","5","6","7","8","9","0"
				};
		for(String aux: charTOremove) {
			str = str.replace(aux, "");
		}
		return str;
	}
	
	String readFromFile(File f){
		try{
			StringBuffer strB = new StringBuffer();
			FileReader fr = new FileReader(f);
			int byteLido;
			while((byteLido = fr.read())!= -1) {
				strB.append((char)byteLido);
			}
			fr.close();
			return strB.toString();
			
		}catch (Exception e) {
			// TODO: handle exception
		}
		return "";
	}
	
	public void moveFile(File origin, File destin){
		try {
			FileInputStream fis = new FileInputStream(origin);
			FileOutputStream fos = new FileOutputStream(destin);
			byte[] buffer = new byte[1024];
			while((fis.read(buffer))!= -1) {
				fos.write(buffer);
				fos.flush();
			}			
			fis.close();
			fos.close();
		} catch (Exception e) {
			// TODO: handle exception
		}
	}
	
	
	public void saveTxtToFile(File outputDir, String fileName, String txt){
		try {			
			File outputFile = new File(outputDir, fileName);
			FileWriter fw = new FileWriter(outputFile, true);
			fw.write(txt);
			fw.flush();
			fw.close();
			
			System.out.println("Extracted: "+outputFile.getName());	
		} catch (Exception e) {
			// TODO: handle exception
		}
	}
	
	public void extractWordsStatistcs(TreeMap<String, Integer> wordStatistics, String txt){
		try {
			String[] splWords = txt.split(" "); 
			for(String word: splWords){
				word = word.trim();
				if(word.isEmpty()) continue;
				if(!wordStatistics.containsKey(word)){
					wordStatistics.put(word, 1);
				}else{
					wordStatistics.put(word, wordStatistics.get(word)+1);
				}				
			}
		} catch (Exception e) {
			// TODO: handle exception
		}
	}
	
	public String radicalize(String str) {
		String[] suffex = new String[] {
				"aca", "eca","ica","oca","uca",
				"aco", "eco","ico","oco","uco",
				"ada", "eda","ida","oda","uda",
				"ado","edo","ido","odo","udo",
				"ando","endo","indo","ondo","undo",
				"ato","eto","ito","oto","uto",
				"anto","ento","into","onto","unto"
				};
		for(String suffix: suffex) {
			String aux = str.endsWith(suffix+"s") || str.endsWith(suffix) ? str.replace(suffix+"s", "").replace(suffix, "") : str;
			if(!aux.equals(str)) {
				str = aux;
				break;
			}
		}
		return str;
	}
	
	public String extractFromPdf(File loadFile,TreeSet<String> stopwords){
		try {
			PDDocument pdf = PDDocument.load(loadFile);
			PDFTextStripper stripper = new PDFTextStripper();
			String txt = stripper.getText(pdf);
			pdf.close();
			
			StringBuffer strB = new StringBuffer();
			String[] lines = removeTrash(txt.toLowerCase()).split("\n");
			
			for(String line: lines) {
				String[] tokens = line.replace(" ","-").split("-");
				for(String token: tokens) {
					token = token.trim();
					if(token.isEmpty())continue;
					if(stopwords.contains(token)) continue;
					if(token.length() < 3 || token.length() > 20)continue;
					strB.append(radicalize(token).concat(" "));	
				}
			}
			
			return strB.toString();
		} catch (Exception e) {
		}
		return "";
	}
	
    public Instances loadWekaDataSource() throws Exception {        
    	DataSource ds = new DataSource(getTrainedFile().getAbsolutePath());
    	Instances instancias = ds.getDataSet();
        instancias.setClassIndex(instancias.numAttributes() - 1);
        return instancias;
    }
}