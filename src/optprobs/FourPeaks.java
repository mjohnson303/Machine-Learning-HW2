package optprobs;

import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class FourPeaks {
    /** The n value */
    private static final int N = 1000;
    /** The t value */
    private static final int T = N / 10;
    
    public static void main(String[] args) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
//      RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
//      FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
//      fit.train();
//      System.out.println(ef.value(rhc.getOptimal()));
      
      FixedIterationTrainer fit;
      for(int i = 0; i<10; i++)
      {
      	System.out.println(i);
	        SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
	        fit = new FixedIterationTrainer(sa, 50);
	        fit.train();
	        System.out.println(ef.value(sa.getOptimal()));
	        
	        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 25, gap);
	        fit = new FixedIterationTrainer(ga, 50);
	        fit.train();
	        System.out.println(ef.value(ga.getOptimal()));
	        
	        MIMIC mimic = new MIMIC(200, 100, pop);
	        fit = new FixedIterationTrainer(mimic, 50);
	        fit.train();
	        System.out.println(ef.value(mimic.getOptimal()));
      }
    }
}
