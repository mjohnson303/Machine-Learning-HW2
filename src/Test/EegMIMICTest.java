package Test;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;

import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.DataSet;
import shared.FixedIterationTrainer;
import shared.GradientErrorMeasure;
import shared.Instance;
import shared.SumOfSquaresError;

/**
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class EegMIMICTest {
    /** The n value */
    private static final int N = 50;
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
    	ArrayList<ArrayList<double[]>>data = new ArrayList<ArrayList<double[]>>();
        ArrayList<Double> labels = new ArrayList<Double>();
        
        try {
			BufferedReader br = new BufferedReader(new FileReader("data/EEG_NN.data"));
			
			String curline = br.readLine();
			String[] splitCurline;
			double[] nextData;
			
			while (curline != null) {
				splitCurline = curline.split(",");
				nextData = new double[splitCurline.length - 2];
				for (int i = 0; i <= nextData.length - 2; i++)
					nextData[i] = Double.parseDouble(splitCurline[i]);
				data.add(new ArrayList<double[]>());
				data.get(data.size() - 1).add(nextData);
				/*data.get(data.size() - 1).add(
						new double[]{
								Double.parseDouble(
										splitCurline[splitCurline.length - 1])});*/
				if(splitCurline[splitCurline.length-1].equals("Open"))
	            	labels.add(1.0);
	            else
	            	labels.add(0.0);
				//labels.add(splitCurline[splitCurline.length-1]);
				curline = br.readLine();
			}
						
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
        System.out.println("EEG");
        Instance[] patterns = new Instance[data.size()];
        for (int i = 0; i < patterns.length; i++) {
            patterns[i] = new Instance(data.get(i).get(0));
            patterns[i].setLabel(new Instance(labels.get(i)));
            //patterns[i].setLabel(new Instance(data.get(i).get(1)));
        }
        BackPropagationNetworkFactory factory = 
                new BackPropagationNetworkFactory();       
        BackPropagationNetwork network = factory.createClassificationNetwork(
           new int[] { 14, 5, 1});
        DataSet d = new DataSet(patterns);
        GradientErrorMeasure errorMeasure = new SumOfSquaresError();
        // for rhc, sa, and ga we use a permutation based encoding
        //NeuralNetworkEvaluationFunction ef = new NeuralNetworkEvaluationFunction(network, d,errorMeasure);
        MyNNEval ef = new MyNNEval(network, d,errorMeasure);
        
        
        Distribution odd = new DiscretePermutationDistribution(14);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        //CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
        CrossoverFunction cf = new UniformCrossOver();
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
//        
//        //FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
////        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 1000);
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 10);
//        fit.train();
//        System.out.println("RHC: "+ef.value(rhc.getOptimal()));
//        
//        SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .95, hcp);
////        fit = new FixedIterationTrainer(sa, 200000);
//        fit = new FixedIterationTrainer(sa, 10);
//        fit.train();
//        //System.out.println("SA: "+ef.value(sa.getOptimal()));
//        
//        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 20, gap);
//        fit = new FixedIterationTrainer(ga, 1000);
//        fit.train();
//        System.out.println(ef.value(ga.getOptimal()));
//        
        // for mimic we use a sort encoding
        double[] trainTime = new double[100];
        double[] testTime = new double[100];
        for(int i = 0; i<100; i++)
        {
        	BackPropagationNetwork network1 = factory.createClassificationNetwork(
        	           new int[] { 14, 5, 1});
	        NeuralNetworkEvaluationFunction ef2 = new NeuralNetworkEvaluationFunction(network1, d,errorMeasure);
	        int[] ranges = new int[N];
	        Arrays.fill(ranges, N);
	        odd = new  DiscreteUniformDistribution(ranges);
	        Distribution df = new DiscreteDependencyTree(.1, ranges); 
	        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef2, odd, df);
	        
	        MIMIC mimic = new MIMIC(200, 100, pop);
	        fit = new FixedIterationTrainer(mimic, 100);
	        double start = System.nanoTime();
	        fit.train();
	        double end = System.nanoTime();
	        trainTime[i]=end-start;
	        start = System.nanoTime();
	        System.out.println(ef.value(mimic.getOptimal()));
	        end=System.nanoTime();
	        testTime[i]=end-start;
        }
        for(int j = 0; j<100; j++)
        	System.out.println(trainTime[j]+" "+testTime[j]);

        
    }
}
