import com.cloudera.sparkts.models.ARIMA;
import com.cloudera.sparkts.models.ARIMAModel;
import org.apache.commons.lang.ArrayUtils;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import pl.zankowski.iextrading4j.api.refdata.ExchangeSymbol;
import pl.zankowski.iextrading4j.api.stocks.Chart;
import pl.zankowski.iextrading4j.api.stocks.ChartRange;
import pl.zankowski.iextrading4j.client.IEXTradingClient;
import pl.zankowski.iextrading4j.client.IEXApiClient;
import pl.zankowski.iextrading4j.client.rest.request.refdata.SymbolsRequestBuilder;
import pl.zankowski.iextrading4j.client.rest.request.stocks.ChartRequestBuilder;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.HashMap;


class Company_IEX{
    private Long id;
    private String symbol;
    private String name;
    Company_IEX(Long i,String sym,String na){
        this.symbol=sym;
        this.name=na;
        this.id=i;
    }
    void showNameAndId(){
        System.out.println(this.name);
    }
    String getSymbol(){
        return this.symbol;
    }
    String getName(){
        return this.name;
    }
    Long getId(){
        return this.id;
    }
}

class Loading_data{
    static ArrayList<Company_IEX> load_company(){
        IEXApiClient iexTradingClient = IEXTradingClient.create();
        List<ExchangeSymbol> exchangeSymbolList = iexTradingClient.executeRequest(new SymbolsRequestBuilder().build());
        ArrayList<Company_IEX> Companies = new ArrayList<Company_IEX>();
        exchangeSymbolList.forEach(x->Companies.add(new Company_IEX(x.getIexId(),x.getSymbol(),x.getName())));
        return Companies;
    }
    static ArrayList<Double> load_stoks(String symbol, String code){
        final IEXApiClient iexTradingClient = IEXTradingClient.create();
        final List<Chart> chartList = iexTradingClient.executeRequest(new ChartRequestBuilder()
                .withChartRange(ChartRange.getValueFromCode(code))
                .withSymbol(symbol)
                .build());
        ArrayList<Double>ans = new ArrayList<Double>();
        chartList.forEach(x->ans.add(x.getClose().doubleValue()));
        return ans;
    }
}


class user_int{
    private static user_int instance = null;

    public static synchronized user_int getInstance() {
        if (instance == null)
            instance = new user_int();
        return instance;
    }
    static void choose_company() throws IOException {
        List<Company_IEX> ciex= Loading_data.load_company();
        ciex.forEach(x->x.showNameAndId());
        Scanner in = new Scanner(System.in);
        System.out.println("Выберите компанию, по которой будет построен прогоз");
        Company_IEX co=null;
        while(co==null){
            String comp = in.nextLine();
            for (Company_IEX x:ciex){
                if(x.getName().equals(comp)){
                    co=new Company_IEX(x.getId(),x.getSymbol(),x.getName());
                    break;
                }
            }
            if(co==null){
                System.out.println("Компании нет, попробуйте еще раз");
            }
        }
        String symbol=co.getSymbol();
        System.out.println("Выберите временной промежуток, по которому строится прогноз");
        HashMap<String,String> times= new HashMap<String, String>();
        times.put("5y","5 лет");
        times.put("2y","2 года");
        times.put("1y","1 год");
        times.put("ytd","Текущий год");
        times.put("6m","Шесть месяцев");
        times.put("3m","3 месяца");
        times.put("1m","1 месяц");
        System.out.println(times.entrySet());
        String period = in.nextLine();
        while (!times.containsKey(period)){
            System.out.println("Значения нет, попробуйте еще раз");
            period=in.nextLine();
        }
        ArrayList<Double>dataDouble = Loading_data.load_stoks(symbol.toLowerCase(),period);
        if(!dataDouble.isEmpty()){
            double[] dataTS=ArrayUtils.toPrimitive(dataDouble.toArray(new Double[dataDouble.size()]));
            dataDouble.clear();
            Model my=new Model(dataTS);
            my.savePicture("predicion_chart.jpeg");
        }else{
            System.out.println("Для данной компании нет биржевых котировок");
        }


    }
}

class Model{
    double[] dataArray;

    Model(double[] ar){
        this.dataArray=ar;
    }

    double[] getParametr(){

        Vector cur= Vectors.dense(dataArray);
        ARIMAModel arimaModel= ARIMA.autoFit(cur,100,100,100);
        System.out.println(arimaModel.p());
        System.out.println(arimaModel.d());
        System.out.println(arimaModel.q());
        return arimaModel.forecast(cur,10).toArray();
    }

    void getArray(){
        for(int i=0;i<dataArray.length;i++){
            System.out.println(dataArray[i]);
        }
    }
    void savePicture(String filename) throws IOException {
        XYSeries da=new XYSeries("Chart");
        double[]p1=getParametr();
        writeToTxt("prediction.txt");
        for (int i = 0; i < p1.length; i++) {
            da.add(i,p1[i]);
        }
        XYDataset xyDataset=new  XYSeriesCollection(da);
        JFreeChart chart=ChartFactory.createXYLineChart("sfsdf","sf","drgf",xyDataset, PlotOrientation.VERTICAL,true,false,false);
        int width=1500;
        int height=1000;
        File lineChart = new File( filename );
        ChartUtilities.saveChartAsJPEG(lineChart ,chart, width ,height);
    }
    void writeToTxt (String filePathName) throws IOException{
        BufferedWriter writer = new BufferedWriter(new FileWriter(new File(filePathName)));
        for (int i = 0; i < this.dataArray.length; i++) {
            writer.write(String.valueOf(this.dataArray[i]));
            writer.write("\n");
        }
        writer.flush();
    }

}



public class Main {
    public static void main(String[] args)throws IOException {
        user_int.choose_company();
    }
}

