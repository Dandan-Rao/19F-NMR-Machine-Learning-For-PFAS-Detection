import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import java.util.Vector;
import java.util.Map;
import org.openscience.cdk.qsar.DescriptorValue;

import org.openscience.cdk.ChemFile;
import org.openscience.cdk.aromaticity.CDKHueckelAromaticityDetector;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IChemFile;
import org.openscience.cdk.interfaces.IChemObject;
import org.openscience.cdk.interfaces.IMolecule;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.io.ISimpleChemObjectReader;
import org.openscience.cdk.io.ReaderFactory;
import org.openscience.cdk.qsar.DescriptorEngine;
import org.openscience.cdk.qsar.IDescriptor;
import org.openscience.cdk.qsar.IMolecularDescriptor;
import org.openscience.cdk.qsar.IAtomicDescriptor;
import org.openscience.cdk.qsar.result.DoubleArrayResult;
import org.openscience.cdk.qsar.result.DoubleArrayResultType;
import org.openscience.cdk.qsar.result.DoubleResult;
import org.openscience.cdk.qsar.result.IDescriptorResult;
import org.openscience.cdk.qsar.result.IntegerArrayResult;
import org.openscience.cdk.qsar.result.IntegerArrayResultType;
import org.openscience.cdk.qsar.result.IntegerResult;
import org.openscience.cdk.tools.manipulator.AtomContainerManipulator;
import org.openscience.cdk.tools.manipulator.ChemFileManipulator;
import org.openscience.cdk.qsar.descriptors.molecular.IPMolecularLearningDescriptor;
import org.openscience.cdk.qsar.descriptors.atomic.IPAtomicLearningDescriptor;
import org.openscience.cdk.smiles.SmilesGenerator;
import org.openscience.cdk.tools.HOSECodeGenerator;
import org.openscience.cdk.exception.CDKException;

import javax.vecmath.Point3d;
import java.util.Scanner;

/**
 * ApplyCDKDescriptors.java
 * Purpose: Calculate CDK descriptors in CSV format from input SDF files
 * For ease of use, the design is completely static, i.e. no member functions
 * Calling the constructor executes the algorithm
 *
 * @author Martin Guetlein, Andreas Maunz
 * @version 1.0 20/9/2012
 */
public class GetCDKDescriptors {
	private static DescriptorEngine ENGINE = new DescriptorEngine(DescriptorEngine.ATOMIC);

  /**
  * Example main
  *
  */
  public static void main(String[] args) throws java.io.IOException {
      // Get the input code from arguments
      if (args.length < 1) {
          System.out.println("Usage: java GetCDKDescriptors <input>");
          System.exit(1);
      }
      String inputCode = args[0]; // First argument passed from Python

      // Array of file names
      String[] fileNames = new String[]{inputCode};

      // Base directory path
      String baseDir = ".." + File.separator + ".." + File.separator + ".." + File.separator + "artifacts" + File.separator + "temp" + File.separator;
      System.out.println(baseDir);
      for (String fileName : fileNames){
          String inpath = baseDir + fileName + ".sdf";
          String outpath = baseDir + fileName + "_Descriptors" + ".csv";
          String outpath2 = baseDir + fileName + "_Neighbors" + ".csv";
          // Create CDK Descriptor.csv from sdf files.
          getDescriptorCSV(inpath, outpath, "");
          // Create spatially neighboring atoms list
          getNearestAtoms(inpath);
          writeNearestAtomsToCSV(inpath, outpath2);
      }
      // Array of file names (without the .sdf extension)
//      String[] fileNames = new String[]{"From_Review_202"};
//
//      // Base directory path
//      String baseDir = "...""
//
//      for (String fileName : fileNames) {
//          String inpath = baseDir + fileName + ".sdf";
//          String outpath = baseDir + fileName + "_Descriptors" + ".csv";
//          String outpath2 = baseDir + fileName + "_Neighbors" + ".csv";
//          // Creat CDK Descriptor.csv from sdf files.
//          getDescriptorCSV(inpath, outpath, "");
//          // Create spatially neighboring atoms list
//          getNearestAtoms(inpath);
//          writeNearestAtomsToCSV(inpath, outpath2);
//      }
  }

    /**
  * Constructor, executing the algorithm
  *
  * @param: string The path to the input SDF file
  * @param: string The path to the output CSV file
  */
  public GetCDKDescriptors(String inpath, String outpath, String descNamesStr) throws java.io.IOException  {
    getDescriptorCSV(inpath,outpath,descNamesStr);
  }



 /**
  * Calculate descriptors. Omits IPMolecularLearningDescriptor
  *
  * @param csvOutputPath path to SDF input file
  * @param csvOutputPath path to CSV output file
  * @param descNamesStr comma-seperated list of descriptor names (if empty, all descriptors will be calculated)
  */
 public static void getDescriptorCSV(String sdfInputPath, String csvOutputPath, String descNamesStr) throws java.io.IOException
 {
     List<IMolecule> mols = readMolecules(sdfInputPath);
     System.err.println("read " + mols.size() + " compounds");  //Output the number of compounds read in
     List<IDescriptor> descriptors = ENGINE.getDescriptorInstances(); // All 30 descriptors
     System.err.println("found " + descriptors.size() + " descriptors");  // Output the number of descriptors found


     List<String> descNames = Arrays.asList(descNamesStr.split(","));  //list of target descriptor

     ArrayList<String> colNames = new ArrayList<String>();
     ArrayList<Double[]> values = new ArrayList<Double[]>();

     //Handle 30 atomic descriptors
     for (IDescriptor desc : descriptors) {   // Traverse all CDK descriptors
         if (desc instanceof IPAtomicLearningDescriptor)    // If desc is instance of IPAtomicLearningDescriptor, skip current iteration
             continue;

         // desc.toString() will output value such as org.openscience.cdk.qsar.descriptors.atomic.VdWRadiusDescriptor@49c386c8
         String tname = desc.getClass().getName();  //Get class name, like org.openscience.cdk.qsar.descriptors.atomic.VdWRadiusDescriptor
         String[] tnamebits = tname.split("\\."); // Seperate using '.'
         tname = tnamebits[tnamebits.length-1]; // Get the last class name VdWRadiusDescriptor

         // 如果存在descNamesStr参数。则
         if ((descNamesStr.length()>0) && (!descNames.contains(tname))) // If descNamesStr has value and contains tname, skip current iteration.
             continue;

         String[] originalColNames = desc.getDescriptorNames(); // These are not class names but rather the names of
                                                            // the individual descriptors that the object can calculate, such as VdWRadius.
         String[] newColNamesArr = new String[originalColNames.length]; // Create a new array for modified names
         for (int idx=0; idx<originalColNames.length; idx++) {
             newColNamesArr[idx] = tname + "-" + originalColNames[idx];
         }


         colNames.addAll(Arrays.asList(newColNamesArr));

         for (IAtomContainer mol : mols) {  // traverse each molecule
             int atomCount = mol.getAtomCount();
             List<IAtom> atoms = new ArrayList<IAtom>();
             for (int i = 0; i < atomCount; i++) {  //traverse each atom
                 atoms.add(mol.getAtom(i));
             }
             values.addAll(computeListsAtomic(mol, atoms, (IAtomicDescriptor) desc)); // Calculate descriptors, then convert values to a list
         }
     }

     int ncol = values.size();  //number of columns
     int nrow = mols.get(0).getAtomCount();   // number of rows
     FileWriter fstream = new FileWriter(csvOutputPath);   //create a new file
     BufferedWriter out = new BufferedWriter(fstream);
     out.write("SMILES,");
     for (int c=0; c<ncol; c++) {
         if (c!=0) out.write(",");
         out.write(colNames.get(c));
     }
     out.write("\n");
     for (int r=0; r<nrow; r++) {
         IAtom atom = mols.get(0).getAtom(r);
         String smi = atom.getSymbol();
         out.write(smi + ",");
         for (int c=0; c<ncol; c++) {
             if (c!=0) out.write(",");
             out.write(""+values.get(c)[r]);
         }
         out.write("\n");
     }
     out.flush();  // write values in "values" to the new file
 }


 public static ArrayList<Double[]> getAtomicDescriptor(String sdf, String descNamesStr) throws java.io.IOException  
 {
    List<IMolecule> mols = readMoleculesString(sdf);
    System.err.println("read " + mols.size() + " compounds");
    List<IDescriptor> descriptors = ENGINE.getDescriptorInstances();
    System.err.println("found " + descriptors.size() + " descriptors");

    List<String> descNames = Arrays.asList(descNamesStr.split(","));
    ArrayList<String> colNames = new ArrayList<String>();
    ArrayList<Double[]> values = new ArrayList<Double[]>();
    for (IDescriptor desc : descriptors) {
      if (desc instanceof IPAtomicLearningDescriptor)
        continue;
      String tname = desc.getClass().getName();
      String[] tnamebits = tname.split("\\.");
      tname = tnamebits[tnamebits.length-1];
      if ((descNamesStr.length()>0) && (!descNames.contains(tname)))
        continue;
      String[] colNamesArr = desc.getDescriptorNames();
      for (int idx=0; idx<colNamesArr.length; idx++) {
        colNamesArr[idx] = tname + "-" + colNamesArr[idx];
      }
      colNames.addAll(Arrays.asList(colNamesArr));
      for (IAtomContainer mol : mols) {
        int atomCount = mol.getAtomCount();
        List<IAtom> atoms = new ArrayList<IAtom>();
        for (int i = 0; i < atomCount; i++) {
          atoms.add(mol.getAtom(i));
        }
        values.addAll(computeListsAtomic(mol, atoms, (IAtomicDescriptor) desc));
        getHoseCodesForMolecule(mol);

        // for (int i = 0; i < atomCount; i++) {
        //   System.out.println(mol.getAtom(i).getSymbol());
        // }
      }
    }

    return values;
 }

 // Find all relevant Fluorine atoms in molecule
public static ArrayList<String> getFluorineAtoms(String sdf) {
  List<IMolecule> mols = readMoleculesString(sdf); // Get molecule
  ArrayList<String> Fluorine_positions = new ArrayList<String>();

  for (IAtomContainer mol : mols) {
    int atomCount = mol.getAtomCount();
    for (int i = 0; i < atomCount; i++) {   // Traverse atoms
      IAtom atom = mol.getAtom(i);

        /**
         * this code is used to go through the atoms in a molecule and find all Fluorine atoms that are not connected to Oxygen or Nitrogen atoms.
         * For each such Fluorine atom found, its position in the molecule is added to a list named fluorine_positions.
         */
      if (atom.getSymbol().equals("F")) {
        List<IAtom> connected_atoms = mol.getConnectedAtomsList(atom);    //retrieves all the atoms that are directly connected to
                                                                    // the Fluorine atom and stores them in a list called connected_atoms.
        if (!connected_atoms.get(0).getSymbol().equals("O") && !connected_atoms.get(0).getSymbol().equals("N")) {   //This part checks the symbol
                                                                                                // of the first atom in the list of connected atoms.
                                                                                // It ensures that this atom is neither Oxygen ("O") nor Nitrogen ("N").
          Fluorine_positions.add(String.valueOf(i));
        }
      }
    }
  }
  return Fluorine_positions;
}

// Find nearest atom to all atoms in a molecule
public static ArrayList<ArrayList<String>> getNearestAtoms(String sdfInputPath) {
//  List<IMolecule> mols = readMoleculesString(sdf);
  List<IMolecule> mols = readMolecules(sdfInputPath);
  ArrayList<ArrayList<String>> atom_distances = new ArrayList<ArrayList<String>>();

  for (IAtomContainer mol : mols) {
    int atomCount = mol.getAtomCount();
    List<IAtom> atoms = new ArrayList<IAtom>();

    // Get all atoms
    for (int i = 0; i < atomCount; i++)
    {
      atoms.add(mol.getAtom(i));
    }
    for (int i = 0; i < atoms.size(); i++)
    {
      Double[] distances = new Double[atoms.size()];
      // Double min_distance = 999999.9999;
      // int min_d_index = -1;
      for (int j = 0; j < atoms.size(); j++)
      {
        if (j == i) {
            // Set the distance between atom and itself to be the largest
          // Large number so that sorting puts it last
          distances[j] = Double.MAX_VALUE;
          continue;
        }
        // a process where distances between pairs of atoms in a 3D space are being calculated and stored.
        Point3d firstPoint = atoms.get(i).getPoint3d();
        Point3d secondPoint = atoms.get(j).getPoint3d();

          // Check if secondPoint is null
//          if (secondPoint == null) {
//              // Handle the case where the Point3d is null (e.g., print an error message)
//              System.err.println("Point3d for atoms[" + j + "] is null.");
//              continue; // Skip this pair of atoms
//          }

        Double distance;
          distance = secondPoint.distance(firstPoint);
          distances[j] = distance;

        // if (distance < min_distance) {
        //   min_distance = distance;
        //   min_d_index = j;
        // }
      }

      //atom_distances.add(String.valueOf(min_d_index));
      ArrayList<String> indices = new ArrayList<String>();
      Double[] d = distances.clone();
      Arrays.sort(d);
      List<Double> d_list = Arrays.asList(distances);

      for (int j = 0; j < distances.length; j++)
      {
        String index = String.valueOf(d_list.indexOf(d[j]));
        indices.add(index);
      }
      atom_distances.add(indices);
    }
  }
  return atom_distances;
}

    public static void writeNearestAtomsToCSV(String sdfInputPath, String csvOutputPath) throws IOException {
        // Call getNearestAtoms to get the data
        ArrayList<ArrayList<String>> atomDistances = getNearestAtoms(sdfInputPath);

        // Open a file writer
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(csvOutputPath))) {
            // Iterate over each list of nearest atoms
            for (ArrayList<String> distances : atomDistances) {
                // Convert the list of strings to a single comma-separated string
                String line = String.join(",", distances);
                // Write this line to the CSV file
                writer.write(line);
                writer.newLine(); // Move to the next line
            }
        }
        // BufferedWriter is automatically closed by the try-with-resources block
    }

public static ArrayList<String> getHoseCodesForMolecule(IAtomContainer mol) {
  HOSECodeGenerator hoseG = new HOSECodeGenerator();
  ArrayList<String> hoseCodes = new ArrayList<String>();

  int atomCount = mol.getAtomCount();
  List<IAtom> atoms = new ArrayList<IAtom>();
  for (int i = 0; i < atomCount; i++) {
    try {
      String hose = hoseG.getHOSECode(mol, mol.getAtom(i), 0);
      hoseCodes.add(hose);
      System.out.println("HOSE = " + hose + "\n");
    }
    catch (CDKException e) {
      e.printStackTrace();
    }
  }
  return hoseCodes;
}

// Calculate RDF Proton descriptors by calculating aromaticity

// public static ArrayList<String> calculateRDFProtonDescriptor(IAtomContainer molecule, Descriptor desc) {
//   ElectronDonation model       = ElectronDonation.cdk();
//   CycleFinder      cycles      = Cycles.cdkAromaticSet();
//   Aromaticity      aromaticity = new Aromaticity(model, cycles);

//   // apply our configured model to each molecule, the CDK model
//   // requires that atom types are perceived
//   AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(molecule);
//   boolean aromaticity = aromaticity.apply(molecule);

//   int atomCount = mol.getAtomCount();
//   List<IAtom> atoms = new ArrayList<IAtom>();

//   for (int i = 0; i < atomCount; i++) {
//     atoms.add(mol.getAtom(i));
//   }
//   // check for RDF descriptors
//   Array<String> value_desc = new Array<String>();
//   for (int i = 0; i < atoms.size(); i++) {
//     value_desc.add(Double.valueOf(descriptor.calculate(molecule, atoms.get(i), aromaticity));
//   }
// }


 /**
  * Get SMILES code for a molecule
  *
  * @param m The molecule
  * @return string The SMILES code
  */
 public static String getSmiles(IMolecule m)
 {
   Map<Object, Object> props = m.getProperties();
   for (Object key : props.keySet()) {
     if (key.toString().equals("STRUCTURE_SMILES") || key.toString().equals("SMILES"))
       return props.get(key).toString();
   }
   SmilesGenerator g = new SmilesGenerator();
   return g.createSMILES(m);
 }


	public static List<Double[]> computeLists(List<IMolecule> mols, IMolecularDescriptor desc )
	{
    System.out.println("computing descriptor " + getName(desc));
    List<Double[]> values = computeDescriptors(mols, (IMolecularDescriptor) desc);
    return values;
	}

  public static List<Double[]> computeListsAtomic(IAtomContainer mol, List<IAtom> atoms, IAtomicDescriptor desc )
  {
    System.out.println("computing descriptor " + getName(desc));
    List<Double[]> values = computeDescriptorsAtomic(mol, atoms, desc);
    return values;
  }


 /**
  * Read in molecules, using any supported format
  *

  * @return Vector<IMolecule> The molecules
  */
	public static List<IMolecule> readMolecules(String filepath)
	{
		Vector<IMolecule> mols = new Vector<IMolecule>();
		File file = new File(filepath);
		if (!file.exists())
			throw new IllegalArgumentException("file not found: " + filepath);
		List<IAtomContainer> list;
		try
		{
			ISimpleChemObjectReader reader = new ReaderFactory().createReader(new InputStreamReader(
					new FileInputStream(file)));
			if (reader == null)
				throw new IllegalArgumentException("Could not determine input file type");
			IChemFile content = (IChemFile) reader.read((IChemObject) new ChemFile());
			list = ChemFileManipulator.getAllAtomContainers(content);
			reader.close();
		}
		catch (Exception e)
		{
			e.printStackTrace();
			return null;
		}

		for (IAtomContainer iAtomContainer : list)
		{
			IMolecule mol = (IMolecule) iAtomContainer;
			mol = (IMolecule) AtomContainerManipulator.removeHydrogens(mol);
			try
			{
				AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(mol);
			}
			catch (Exception e)
			{
				e.printStackTrace();
			}
			try
			{
				CDKHueckelAromaticityDetector.detectAromaticity(mol);
			}
			catch (Exception e)
			{
				e.printStackTrace();
			}
			if (mol.getAtomCount() == 0)
				System.err.println("molecule has no atoms");
			else
				mols.add(mol);
		}
		return mols;
	}

  public static List<IMolecule> readMoleculesString(String sdf)
  {
    Vector<IMolecule> mols = new Vector<IMolecule>();
    if (sdf.equals(""))
      throw new IllegalArgumentException("No sdf found" + sdf);
    List<IAtomContainer> list;
    try
    {
      InputStream is = new ByteArrayInputStream(sdf.getBytes("UTF-8"));
      ISimpleChemObjectReader reader = new ReaderFactory().createReader(new InputStreamReader(is));
      if (reader == null)
        throw new IllegalArgumentException("Could not determine input file type");
      IChemFile content = (IChemFile) reader.read((IChemObject) new ChemFile());
      list = ChemFileManipulator.getAllAtomContainers(content);
      reader.close();
    }
    catch (Exception e)
    {
      e.printStackTrace();
      return null;
    }

    for (IAtomContainer iAtomContainer : list)
    {
      IMolecule mol = (IMolecule) iAtomContainer;
      //mol = (IMolecule) AtomContainerManipulator.removeHydrogens(mol);
      try
      {
        AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(mol);
      }
      catch (Exception e)
      {
        e.printStackTrace();
      }
      try
      {
        CDKHueckelAromaticityDetector.detectAromaticity(mol);
      }
      catch (Exception e)
      {
        e.printStackTrace();
      }
      if (mol.getAtomCount() == 0)
        System.err.println("molecule has no atoms");
      else
        mols.add(mol);
    }
    return mols;
  }

	public static List<Double[]> computeDescriptors(List<IMolecule> mols, IMolecularDescriptor descriptor)
	{
		List<Double[]> vv = new ArrayList<Double[]>();

		for (int j = 0; j < getSize(descriptor); j++)
			vv.add(new Double[mols.size()]);

		for (int i = 0; i < mols.size(); i++)
		{
			if (mols.get(i).getAtomCount() == 0)
			{
				for (int j = 0; j < getSize(descriptor); j++)
					vv.get(j)[i] = null;
			}
			else
			{
				try
				{
					IDescriptorResult res = descriptor.calculate(mols.get(i)).getValue();
					if (res instanceof IntegerResult)
						vv.get(0)[i] = (double) ((IntegerResult) res).intValue();
					else if (res instanceof DoubleResult)
						vv.get(0)[i] = ((DoubleResult) res).doubleValue();
					else if (res instanceof DoubleArrayResult)
						for (int j = 0; j < getSize(descriptor); j++)
							vv.get(j)[i] = ((DoubleArrayResult) res).get(j);
					else if (res instanceof IntegerArrayResult)
						for (int j = 0; j < getSize(descriptor); j++)
							vv.get(j)[i] = (double) ((IntegerArrayResult) res).get(j);
					else
						throw new IllegalStateException("Unknown idescriptor result value for '" + descriptor + "' : "
								+ res.getClass());
				}
				catch (Throwable e)
				{
					System.err.println("Could not compute cdk feature " + descriptor);
					e.printStackTrace();
					for (int j = 0; j < getSize(descriptor); j++)
						vv.get(j)[i] = null;
				}
			}
			for (int j = 0; j < getSize(descriptor); j++)
				if (vv.get(j)[i] != null && (vv.get(j)[i].isNaN() || vv.get(j)[i].isInfinite()))
					vv.get(j)[i] = null;
		}

		return vv;
	}

  public static List<Double[]> computeDescriptorsAtomic(IAtomContainer mol, List<IAtom> atoms, IAtomicDescriptor descriptor)
  {
    List<Double[]> vv = new ArrayList<Double[]>();
    vv.add(new Double[atoms.size()]);
    for (int i = 0; i < atoms.size(); i++)
    {
      if (atoms.get(i) == null)
      {
          vv.get(0)[i] = null;
      }
      else
      {
        try
        {
          IDescriptorResult res = descriptor.calculate(atoms.get(i), mol).getValue();
          if (res instanceof IntegerResult) {
            vv.get(0)[i] = (double) ((IntegerResult) res).intValue();
            //System.out.println(vv.get(0)[i]); 
          }
          else if (res instanceof DoubleResult) {
            vv.get(0)[i] = ((DoubleResult) res).doubleValue();
            //System.out.println(vv.get(0)[i]); 
          }
          else if (res instanceof DoubleArrayResult) {
            vv.get(0)[i] = ((DoubleArrayResult) res).get(0); 
            //System.out.println(vv.get(0)[i]);  
          }
          else if (res instanceof IntegerArrayResult) {
            vv.get(0)[i] = (double) ((IntegerArrayResult) res).get(0);
            //System.out.println(vv.get(0)[i]); 
          }
          else
            throw new IllegalStateException("Unknown idescriptor result value for '" + descriptor + "' : "
                + res.getClass());
        }
        catch (Throwable e)
        {
          System.err.println("Could not compute cdk feature " + descriptor);
          e.printStackTrace();
            vv.get(0)[i] = 0.0;
        }
      }
      if (vv.get(0)[i] != null && (vv.get(0)[i].isNaN() || vv.get(0)[i].isInfinite()))
        vv.get(0)[i] = 0.0;
    }

    return vv;
  }


 /**
  * Get length of result for a given descriptor
  *
  * @param descriptor The descriptor
  * @return int The length
  */
	private static int getSize(IMolecularDescriptor descriptor) 
  {
		IDescriptorResult r = descriptor.getDescriptorResultType();
		if (r instanceof DoubleArrayResultType)
			return ((DoubleArrayResultType) r).length();
		else if (r instanceof IntegerArrayResultType)
			return ((IntegerArrayResultType) r).length();
		else
			return 1;
	}


 /**
  * Get name for a given descriptor
  *
  * @param descriptor The descriptor
  */
	private static String getName(IDescriptor descriptor) 
  {
    try
    {
      String name = ENGINE.getDictionaryTitle(descriptor.getSpecification()).trim();
  		if (name != null) {
        return name;
      }
      else {
        return "";
      }
    }
    catch (Throwable e)
    {
      return "";
    }
	}


}
