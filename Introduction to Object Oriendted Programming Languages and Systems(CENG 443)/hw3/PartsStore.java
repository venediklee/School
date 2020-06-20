import java.awt.print.Printable;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Array;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class PartsStore
{
	private ArrayList<String[]> data = new ArrayList<>();
	
	public PartsStore()
	{
		
		String row = null;
		
		
		BufferedReader csvReader = null;
		try
		{
			csvReader = new BufferedReader(new FileReader("pcparts.csv"));
		}
		catch (FileNotFoundException e)
		{
			System.out.println("can't open file");
		}
		while (true)
		{
			try
			{
				if ((row = csvReader.readLine()) == null)
				{
					break;
				}
			}
			catch (IOException e)
			{
				System.out.println("can't read line");
			}
			
			String[] rowData = row.split(",");
			data.add(rowData);
		}
		try
		{
			csvReader.close();
		}
		catch (IOException e)
		{
			System.out.println("can't close file");
		}
	}
	
	
	private void PrintArray(String[] strings)
	{
		System.out.print(strings[0]);
		for (int i = 1; i < strings.length; i++)
		{
			System.out.print("," + strings[i]);
		}
		System.out.println();
	}
	
	
	public void FindPartsWithBrand(String type, String brand)
	{
		if (type == null)
		{
			data.stream().filter(part -> part[1].equals(brand)).forEach(this::PrintArray);
		}
		else
		{
			data.stream().filter(part -> part[0].equals(type) && part[1].equals(brand)).forEach(this::PrintArray);
		}
	}
	
	public void TotalPrice(String type, String brand, String model)
	{
		Stream<String[]> dataStream = data.stream();
		if (type != null)
		{
			dataStream = dataStream.filter(part -> part[0].equals(type));
		}
		if (brand != null)
		{
			dataStream = dataStream.filter(part -> part[1].equals(brand));
		}
		if (model != null)
		{
			dataStream = dataStream.filter(part -> part[2].equals(model));
		}
		
		System.out.println(new DecimalFormat("#0.00").format(dataStream.mapToDouble(part -> Double.parseDouble(part[part.length - 1].substring(0, part[part.length - 1].length() - 4))).sum()) + " USD");
	}
	
	public void UpdateStock()
	{
		ArrayList<String[]> notStockedItems = new ArrayList<String[]>(data.stream().filter(part -> part[part.length - 1].equals("0.00 USD")).collect(Collectors.toList()));
		System.out.println(notStockedItems.size() + " items removed.");
		data.removeAll(notStockedItems);
	}
	
	public void FindCheapestMemory(int capacity)
	{
		Optional<String[]> resultingMemory = data.stream().filter(part -> part[0].equals("Memory")).
				filter(part -> Integer.parseInt(part[4].substring(0, part[4].length() - 2)) >= capacity).
				min(Comparator.comparing(part -> Double.parseDouble(part[part.length - 1].substring(0, part[part.length - 1].length() - 4))));
		resultingMemory.ifPresent(this::PrintArray);
	}
	
	public void FindFastestCPU()
	{
		Optional<String[]> resultingCPU = data.stream().filter(part -> part[0].equals("CPU")).
				max(Comparator.comparing(part -> Integer.parseInt(part[3]) *
						Double.parseDouble(part[4].substring(0, part[4].length() - 3))));
		resultingCPU.ifPresent(this::PrintArray);
	}
}
