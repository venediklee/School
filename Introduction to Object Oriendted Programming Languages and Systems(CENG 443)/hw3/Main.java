import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.lang.*;


class Main
{
	public static void main(String[] args) throws IOException
	{
		PartsStore a =new PartsStore();
		PartsStore ps = new PartsStore();
		ps.FindPartsWithBrand("Keyboard", "Logitech");
		ps.FindPartsWithBrand(null, "Logitech");
		ps.TotalPrice("GPU", "Asus", "GeForce RTX 2080");
		ps.TotalPrice(null, "Asus", null);
		ps.TotalPrice("GPU", "Asus", null);
		ps.FindCheapestMemory(16);
		ps.FindCheapestMemory(32);
		ps.FindCheapestMemory(64);
		ps.FindFastestCPU();
		ps.UpdateStock();
		ps.FindPartsWithBrand("Keyboard", "Logitech");
		ps.FindPartsWithBrand(null, "Logitech");
		ps.TotalPrice("GPU", "Asus", "GeForce RTX 2080");
		ps.TotalPrice(null, "Asus", null);
		ps.FindCheapestMemory(16);
		ps.FindCheapestMemory(32);
		ps.FindCheapestMemory(64);
		ps.FindFastestCPU();
		ps.UpdateStock();
	}
}