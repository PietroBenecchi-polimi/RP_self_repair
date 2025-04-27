
// per info apri 0README_V6.txt
#include <fstream>
#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <cstring>
#include <filesystem>
#include <algorithm>
//#include <stdin>


namespace fs = std::__fs::filesystem;
using namespace std;
using namespace fs;

const int OUT_c = 0; // per la stampa di controllo
const string DIV =  "////////////////////////////////////////////////";
const string DIVstart =  "///////";
const string DIVend =    "-------";
const string DIVcom =    "*******" ; 
const string FINE_PROG = DIVend+ " END " + DIVend;
const string START_PROG = DIVstart + "START" + DIVstart;
const string UN_OP_FI = "**ERR: unable to open the file:**\n";
const string BB = "/";
const string CLOSE_GLODEC = "</declaration>";
const string CLOSE_LOCDEC = "</declaration>"; 
const string CLOSE_SYSDEC = "</system>";
const string SET_0_IMPSENSE  = "0.0";
const string SET_0_IMPENTITY = "0.0"; 
const string SET_0_SEEAO = "{0,0,0,0,0}";
const string SET_0_OBJ = "{0.0,0.0,0.0}";
const string NR_OBJ = "nr_obj_max";
const string NR_IMP_ENTITY  ="nr_impEntity_max";
const string NR_IMP_SENSE ="nr_impSense_max";
const string NR_SEEAO = "nr_seeao_max";
const string BEFORE_NAME  ="_";//caracter to put before the name give to the template
//per partire a leggere il file csv associato dalla riga giusta 
const int INDEX_SEEAO = 1; // la prima riga è quella dei momi etc...
const int INDEX_IN_IMP_ENTITY = 1;
const int INDEX_IN_IMP_SENSE = 1;
const int INDEX_OBJ = 1; //la prima riga del file csv è per l'intestazione 

const int GLODEC_INX_SENSE = 2;
const int GLODEC_INX_ENTITY = 1;
const int GLODEC_INX_OBJ = 3;
const int GLODEC_INX_SEEAO = 0;
const int INDEX_GEN = 1; //indice di partenza per leggere i file nel file gen
path command_argv;//per poterne aver accesso nella funzione Dpa_argAll_sysDec_main()
string MISSION_FOLDER;//to save the  name of the mission folder
string XML_MAIN_FOLDER;//to save the name of the xml folder
string XML_MAIN_NAME;//to save the name of the xml folder
// per la posizioni delle informazioni seguendo le righe del file config
const int IDX_CONFIG_ENTITY = 0;
const int IDX_CONFIG_SENSE = 1;
const int IDX_CONFIG_SEEAO = 2;
const int IDX_CONFIG_OBJ = 3;
const int IDX_CONFIG_ARGV = 4;

/*HF_config.csv
csv file imp ENTITY name,HF_impEntity.csv
csv file imp SENSE name,HF_impSense.csv
csv file SEEAO name,HF_seeao.csv
csv file OBJ name,HF_obj_env1.csv
csv file ARGV name,HF_arg.csv*/

const int OUT = 0;//per interazione[1], per non interazioni [0]

//function used in the program 
vector<string> vectorStringFromFile(path pr_txt);
void write_to_file(vector<string> vector_string, path pa_txt);
vector<string> add_string(vector<string> vector_string, vector<string> da_agg);
vector<string> f_print_txt_file(path pr_txt);
vector<string> f_vecArg_txt(path argv,string name);
vector<string> ad_GloDec_sta(path pr_txt);// per la parte statica delle global dec utile per il template argAll
vector<string> ad_globDec_dyn_format();//per format da mettere nelle global dec 
vector<string> f_importance_matrix_num_txt_argAll(path pr_csv_entity,path pr_csv_sense, string name);//per il caso argAll
vector<string> matrix_SEEAO_Dpa_argAll(path pr_excel_prova, string name);
vector<string> matrix_obj_xye_argAll(path pr_csv_obj_xye,string name);//per stampare la matrice nel caso della argAll
vector<string> vecArg_ad(path pr_excel_prova,path obj_sta_matr, string name); 
vector<string> f_selection_string_v2_num_argAll(path pr_csv_sense, int glodec_inx, int index,string SET_0);
vector<int> argv_ad_globDec_dyn ;//= ad_globDec_dyn();//funzione per definire le variabili che dipendono dalla grandezza delle matrici/vettori
													  //dei file csv
vector<int> ad_globDec_dyn();

vector<string> f_selection_string_vecArg(path pr_csv);
int csv_rows2col(path obj_sta_matr);//ritorna il valore delle righe di un file csv a due colonne
int csv_n_rows5col(path obj_sta_matr);//numero righe colonne delle righe di un file csv a 5 colonne

string max (vector<string> v_sense, int index);
int max_int (vector<int> v_sense, int index); //stesso della funzione precedente ma sotto forma di intero


int main (int argc, char *argv[]){
	// sviluppo nel main quello che mi serve per stamprare solo in main.xml, senza avere nessuna chiamata ad altre funzioni se non strettamente necessario
	// tenendo almeno quelle basi si scrittura e stampa se servono.
	MISSION_FOLDER = argv[8];// per far in modo che la folder della mission viene cambiata		
	XML_MAIN_FOLDER = argv[9]; // dove voglio generare il main xml 		
	XML_MAIN_NAME  = argv[10];	
	command_argv = MISSION_FOLDER +BB+ argv[1];//file in cui si trovano i nomi dei vari file e il file di configurazione associato l'attuale "0csv_toGen_v1.csv"
	
	argv_ad_globDec_dyn = ad_globDec_dyn();//funzione per definire le variabili che dipendono dalla grandezza delle matrici/vettori
													  //dei file csv dipendendente dal "command argv" saperlo ora mi server per le funzioni matrixSEEAOargAll.. etc 

	//up_argAll_v2 (vector<string> argv,vector<string> argv_c); // prendo quello che avviene in questa funzione
																// e gli do i giusti elementi per porter far funzionare solo la
																// parte che mi interessa togliendo tutto il non necessario

											
	vector<string> vector_string_main;
	stringstream ss;
	
	//global dec 
	vector_string_main = add_string(vector_string_main,f_print_txt_file(argv[2]));//open glob 
	vector_string_main = add_string(vector_string_main,f_print_txt_file(argv[3]));//prof global dec

	ss<<"\n"<<DIVstart<<" static globalDec "<<DIVstart;
	vector_string_main.push_back(ss.str());
	ss.str("");

	vector_string_main = add_string(vector_string_main,ad_GloDec_sta(argv[4]));//my addiction static global dec

	ss<<"\n"<<DIVstart<<" max globalDec "<<DIVstart;
	vector_string_main.push_back(ss.str());
	ss.str("");
	vector_string_main = add_string(vector_string_main,ad_globDec_dyn_format());//addiction of the global constants depending
																					// on the csv data rows
	vector_string_main.push_back(CLOSE_GLODEC); // close Global Declaration 
	
	// system dec 
		vector_string_main = add_string(vector_string_main,f_print_txt_file(argv[5]));//open sys dec
		//vector_string = add_string(vector_string,Dpa_argAll_sysDec_main(argv));// nuova funzione per le sysDec

		// sostituisco il corpo della funzione: 
		//	vector<string> Dpa_argAll_sysDec_main(vector<string> argv){
			// usare la fuzione che fa già questo su uno singolo ed applicarla su tutti i template presenti.
			// ed aggiungere al vector string precedente il vector string definito da questo

		// read the csv file
		ifstream excel_data;
		excel_data.open(absolute(command_argv));

		
		vector<string> vector_string;// vettore dove memorizzo la parte che andrò a scrivere la parte delle system dec
		
		vector<string> v_type,v_name,v_csv;// per memorizzare i valori come stringhe

		// to test the file can be open or not
		if(excel_data.fail()){
			cerr<<"Dpa_argAll_sysDec_main:\t"<<UN_OP_FI << absolute(command_argv)<< endl;
			//return 1;
		}
		// read the file considering the data in that csv file 
		while (excel_data.peek()!=EOF)
		{
			string name,type,csv;
			getline(excel_data,type,',');
			getline(excel_data,name,',');
			getline(excel_data,csv,'\n');
			v_type.push_back(type);
			v_name.push_back(name);
			v_csv.push_back(csv);
		}
		// chiudere il file 
		excel_data.close();

		for(int i = INDEX_GEN; i < v_type.size(); i++){

			// leggere il file csv config per prendere gli argv che mi servono per le funzioni sotto
				//funzione che uso per la generazion del vettore argv: 
				//vector<string> returnArgv(vector<string> arg_static, vector<string> arg_dyn, string name, string AD_TYPE_FOLDER){
				vector<string> arg_dyn = vectorStringFromFile(MISSION_FOLDER+BB+v_name[i]+ BB + v_csv[i]);// ottengo i file che ci sono nel file di configurazione 
				vector<string> argv_1; // sto facendo in questo modo solo per averlo esteso posso fare anche più semplicemente come faccio sotto
				argv_1.push_back(MISSION_FOLDER+BB+ v_name[i]+ BB + arg_dyn[IDX_CONFIG_ENTITY].c_str()) ;//argv_1[9]->0
				argv_1.push_back(MISSION_FOLDER+BB+ v_name[i]+ BB+ arg_dyn[IDX_CONFIG_SENSE].c_str()) ;//argv_1[10]->1
				argv_1.push_back(MISSION_FOLDER+BB+ v_name[i]+ BB + arg_dyn[IDX_CONFIG_SEEAO].c_str()) ;//argv_1[11]->2
				argv_1.push_back(MISSION_FOLDER+BB+ v_name[i]+ BB+arg_dyn[IDX_CONFIG_OBJ].c_str()) ;//argv_1[12]->3
				argv_1.push_back(MISSION_FOLDER+BB+ v_name[i]+ BB+arg_dyn[IDX_CONFIG_ARGV].c_str()) ;//argv_1[13]->4
				argv_1.push_back(BEFORE_NAME + v_name[i].c_str());//argv_1[16]	->5
				//out_vector_string_v1(argv_1,"stampare argv_1?[y/n]",1);
		
			//vector_string = add_string(vector_string,Dpa_argAll_sysDec(argv));
			// sostituisco il corpo della funzione: //vector<string> Dpa_argAll_sysDec(vector<string> argv){
				vector<string> importance_matrix_num = f_importance_matrix_num_txt_argAll(argv_1[IDX_CONFIG_ENTITY],argv_1[IDX_CONFIG_SENSE],argv_1[5]); // per stampare la importance matrix delle entità e dei sensi
				//f_print_txt_file(path pr_txt = "txt_declaration_code.txt");// per creare il vettore stringa di un file di testo
				vector<string> matrix_seeao_num = matrix_SEEAO_Dpa_argAll(argv_1[IDX_CONFIG_SEEAO],argv_1[5]);//devo creare una versione 2 della funzione 
				vector<string> matrix_obj_xye_num =  matrix_obj_xye_argAll(argv_1[IDX_CONFIG_OBJ],argv_1[5]);
				vector<string> init_cases = f_vecArg_txt(argv_1[IDX_CONFIG_ARGV],argv_1[5]);
				vector<string> vecAd = vecArg_ad(argv_1[IDX_CONFIG_SEEAO],argv_1[IDX_CONFIG_OBJ],argv_1[5]);//matrix_caso(argv[13],argv[16]);
				//vector<string> my_code_no_arg = f_print_txt_file(argv[14]);//"txt_hf1_extCodeDpa.xml");
				//vector<string> prof_code_Dpa  = f_print_txt_file(argv[15]);//"txt_hf1_prDecLocDpa.xml");
				//vector<string> vector_string;
				stringstream ssl; 


				vector_string.push_back(DIVstart +" "+ v_name[i] + " "+ DIVstart);
				ssl<<"//args  "<<v_name[i];
				vector_string.push_back(ssl.str());
				ssl.str("");
				vector_string = add_string(vector_string,init_cases);
				vector_string.push_back(DIV);

				vector_string.push_back(DIV);
				ssl<<"//args_ad  "<<v_name[i];
				vector_string.push_back(ssl.str());
				ssl.str("");
				vector_string = add_string(vector_string,vecAd);
				vector_string.push_back(DIV);

				ssl<<"//matrix seeao num "<<v_name[i];
				vector_string.push_back(ssl.str());
				ssl.str("");
				vector_string = add_string(vector_string,matrix_seeao_num);
				vector_string.push_back(DIV);

				ssl<<"//matrix_obj_xye_num "<<v_name[i];
				vector_string.push_back(ssl.str());
				ssl.str("");
				vector_string = add_string(vector_string,matrix_obj_xye_num);
				vector_string.push_back(DIV);

				ssl<<"//importance_matrix_num "<<v_name[i];
				vector_string.push_back(ssl.str());
				ssl.str("");
				vector_string = add_string(vector_string,importance_matrix_num);
				vector_string.push_back(DIV);

			}// for end della  fuzione DpaArgAllmain


		// aggiungo il vettore creato dal for (sono system declaration che dipendono dai file di configurazione)
		vector_string_main = add_string(vector_string_main,vector_string);
		vector_string_main = add_string(vector_string_main,f_print_txt_file(argv[6])); // instances
		vector_string_main.push_back(CLOSE_SYSDEC); // close sysDec 

	// end file 		
		vector_string_main = add_string(vector_string_main,f_print_txt_file(argv[7]));// close file

	write_to_file(vector_string_main, absolute(XML_MAIN_FOLDER +BB+ XML_MAIN_NAME ));//nel file che ho settato nelle global dec

	//f_main_v1(argv[1]);
	cout<<FINE_PROG<<endl;

	return 0;
}




vector<string> add_string(vector<string> vector_string, vector<string> da_agg){

	for(int i =0; i<da_agg.size();i++){
		vector_string.push_back(da_agg[i]);
	}
	return vector_string;
}


vector<string> vectorStringFromFile(path pr_txt){
   //cout <<"ci sto lavorando\n";
	// read the csv file
	ifstream excel_data;
	excel_data.open(absolute(pr_txt));
	// modello stampa
		//const int caso_obj = ...;
	
	vector<string> vector_string;
	
	vector<string> v_column,v_value;// per memorizzare i valori come stringhe
	// to test the file can be open or not
	if(excel_data.fail()){
		cerr<<"vectorStringFromFile\t"<<UN_OP_FI << absolute(pr_txt)<< endl;
		//return 1;
	}

	while (excel_data.peek()!=EOF)
	{
		string column,value;
		getline(excel_data,column,',');
		getline(excel_data,value,'\n');
		v_column.push_back(column);
		v_value.push_back(value);
	}
	

	excel_data.close();

	
	
  
  // enter location ////////////////////////////////////////////////////

	//int location_length = 10;
	int arraySizeSense = v_column.size();
	//int n_rows =  arraySize/n_columns;
	//cout<<n_rows<<endl;
	//cout<<10%n_columns<<endl;
	if(OUT){
	cout << "array size= " << arraySizeSense << endl;
	}
	//vector<string> vector_stampa;
	int index_init = 0;
	// index di partenza, parto da 0 perchè uso tutte le stampe per non considerare la prima riga che ci sono i max
	// e la seconda i nomi delle colonne
	// rapprensenta anche le righe da non contare nella matrice finale
	
	
	for(int i = index_init; i < arraySizeSense;i ++){ 
		stringstream ss;
		ss << v_value[i];

		vector_string.push_back(ss.str());

	} ////////////////////////////////////////////////////////////////////////

	
    return vector_string;
}


vector<string> f_print_txt_file(path pr_txt){
    vector<string> vector_string;
   // path pr_txt = "txt_declaration_code.txt";
	path pa_txt = fs::absolute(pr_txt);
	// read the csv file
	ifstream excel_data;
	excel_data.open(pa_txt);

    	if(excel_data.fail()){
		cerr<<"f_print_txt_file:\t"<<UN_OP_FI << pa_txt<< endl;
		//return 1;
	}

	while (excel_data.peek()!=EOF)
	{
		string x,y,entity;
		getline(excel_data,x);
		//vector_string.push_back(data);
		/*cout<<"sense:"<< sense<<endl;
		cout<<"entity:"<< entity<<endl;
		cout<<"event:"<< event<<endl;
		cout<<"action:"<< action <<endl;
		cout<<"output:"<< output <<endl;
		//cout<<data<<endl;*/
		vector_string.push_back(x.c_str());

	}
	
	excel_data.close();
	
    return vector_string;
}

vector<string> f_selection_string_v2_num(path pr_csv){
path pa_csv = absolute(pr_csv);
ifstream excel_data;
	excel_data.open(pa_csv);
	// modello stampa
	// const int n_columns = 5; // one for each element 
   //const int n_rows = 5;
	//const int SEEAO[n_rows][n_columns]= {{1,1,1,1,1},{2,1,2,2,2},{3,3,3,3,3},{4,4,4,4,4},{5,5,5,5,4}};
	
	vector<string> vector_string;
	vector<string> v_column,v_value,v_name,v_weight;// per memorizzare i valori come stringhe
	// to test the file can be open or not
	if(excel_data.fail()){
		cerr<<"f_selection_string_v2_num:\t"<<UN_OP_FI << pa_csv << endl;
		//return 1;
	}

	while (excel_data.peek()!=EOF)
	{
		string column,value,name,weight;

		getline(excel_data,column,',');
		getline(excel_data,value,',');
		getline(excel_data,name,',');
        getline(excel_data,weight,'\n');
		//vector_string.push_back(data);
		/*cout<<"sense:"<< sense<<endl;
		cout<<"entity:"<< entity<<endl;
		cout<<"event:"<< event<<endl;
		cout<<"action:"<< action <<endl;
		cout<<"output:"<< output <<endl;*/
		//cout<<data<<endl;
		v_column.push_back(column);
		v_value.push_back(value);
		v_name.push_back(name);
        v_weight.push_back(weight);
		
	}
	
	excel_data.close();

	
	// uso dei dati per formato stampa 
	int arraySizeSense = v_column.size();
	if(OUT){
	cout << "array size= " << arraySizeSense << endl;
	}
	//vector<string> vector_stampa;
	stringstream ss1;
	int index_init = 1; // togliere la prima riga perchè rappresentano i nomi delle colonne
	
	for(int i = index_init; i < arraySizeSense-1;i ++){ 
		stringstream ss;
		ss << v_weight[i]<<","; 

		vector_string.push_back(ss.str());
		

	} ////////////////////////////////////////////////////////////////////////
	int i = arraySizeSense-1;
	stringstream ss;
		ss << v_weight[i]<<"\n};"; 

		vector_string.push_back(ss.str());

return vector_string;
}





vector<string> matrix_SEEAO_Dpa_argAll(path pr_excel_prova, string name){
	//cout <<"ci sto lavorando\n";
	// read the csv file
	 std::string::size_type sz;
	ifstream excel_data;
	excel_data.open(absolute(pr_excel_prova));
	// modello stampa
	// const int n_columns = 5; // one for each element 
   //const int n_rows = 5;
	//const int SEEAO[n_rows][n_columns]= {{1,1,1,1,1},{2,1,2,2,2},{3,3,3,3,3},{4,4,4,4,4},{5,5,5,5,4}};
	
	vector<string> vector_string;
	string initialization = "const int SEEAO"+ name + "["+NR_SEEAO+"][nc_seeao]={"; //"const int SEEAO_**HF**[n_rows][n_columns]={"
	string nr_rows ="const int n_rows"+name+" =";
	string nr_col ="const int n_columns"+ name+" =";
	string end = "};" ;
	//vector<int> v_sense_int, v_entity_int,v_action_int, v_output_int,v_event_int; // pper memorizzare i valori come numeri interi
	vector<string> v_sense,v_entity,v_event,v_action, v_output;// per memorizzare i valori come stringhe
	// to test the file can be open or not
	if(excel_data.fail()){
		cerr<<"matrix_SEEAO_Dpa_argAll:\t"<<UN_OP_FI << absolute(pr_excel_prova)<< endl;
		//return 1;
	}

	while (excel_data.peek()!=EOF)
	{
		string data,sense,entity,event,action,output;
		getline(excel_data,sense,',');
		getline(excel_data,entity,',');
		getline(excel_data,event,',');
		getline(excel_data,action,',');
		getline(excel_data,output,'\n');
		//vector_string.push_back(data);
		/*cout<<"sense:"<< sense<<endl;
		cout<<"entity:"<< entity<<endl;
		cout<<"event:"<< event<<endl;
		cout<<"action:"<< action <<endl;
		cout<<"output:"<< output <<endl;
		//cout<<data<<endl;*/
		v_sense.push_back(sense);
		v_entity.push_back(entity);
		v_event.push_back(event);
		v_action.push_back(action);
		v_output.push_back(output);
	}
	/*	for(int i =0;i <v_sense.size();i++)
	{
		v_sense_int.push_back(stoi(v_sense[i].c_str()));
		v_entity_int.push_back(stoi(v_entity[i].c_str()));
		v_event_int.push_back(stoi(v_event[i].c_str()));
		v_action_int.push_back(stoi(v_action[i].c_str()));
		v_output_int.push_back(stoi(v_output[i].c_str()));
	}
	*/
	


	excel_data.close();

	
	
  
  // enter location ////////////////////////////////////////////////////
	int n_columns = 5;
	//int location_length = 10;
	int arraySizeSense = v_sense.size();
	//int n_rows =  arraySize/n_columns;
	//cout<<n_rows<<endl;
	//cout<<10%n_columns<<endl;
	if(OUT){
	cout << "array size= " << arraySizeSense << endl;
	}
	//vector<string> vector_stampa;
	stringstream ss1,ss2;
	//int index_init = 3;
	// parto da 3 per non considerare la prima riga che ci sono i nomi , la 2 il valore dei max
	// e la terza i nomi delle colonne
	// rapprensenta anche le righe da non contare nella matrice finale
	/*
	ss1 <<nr_rows<<arraySizeSense-index_init<<";";
	vector_string.push_back(ss1.str());
	ss2 <<nr_col<<n_columns<<";";
	vector_string.push_back(ss2.str());*/
	vector_string.push_back(initialization);
	
	for(int i = INDEX_SEEAO; i < arraySizeSense-1;i ++){ 
		stringstream ss;
		ss << "{" << v_sense[i] <<"," << v_entity[i]<<","<<v_event[i]<<","<<v_action[i]<<","<<v_output[i]<<"},"  ;

		vector_string.push_back(ss.str());
		

	} ////////////////////////////////////////////////////////////////////////
	
	// aggiunto perchè l'ultimo ha un stringa diversa, non ha la virgola 
	stringstream ss;
	// aggiungere la parte per aggiungere righe che sono pari a zero quando n_rows < nr_seeao_max
	if((v_sense.size()-INDEX_SEEAO) < argv_ad_globDec_dyn[GLODEC_INX_SEEAO] ){
		int i = v_sense.size()-INDEX_SEEAO ;
		ss << "{" << v_sense[i] <<"," << v_entity[i]<<","<<v_event[i]<<","<<v_action[i]<<","<<v_output[i]<<"},"  ;
		vector_string.push_back(ss.str());
		ss.str("");

		for(int j = (v_sense.size()-INDEX_SEEAO); j < argv_ad_globDec_dyn[0]-1; j++){
			ss << SET_0_SEEAO<<",";
			vector_string.push_back(ss.str());
			ss.str("");
		}	
			ss << SET_0_SEEAO;
			vector_string.push_back(ss.str());
			ss.str("");

	}
	else{
		int i = arraySizeSense-1 ;
		ss << "{" << v_sense[i] <<"," << v_entity[i]<<","<<v_event[i]<<","<<v_action[i]<<","<<v_output[i]<<"}"  ;
		vector_string.push_back(ss.str());
		ss.str("");
	}

	vector_string.push_back(end);//fine stampa della matrice

	int arraySize = vector_string.size();
  	
	return vector_string;
}

 
vector<string> vecArg_ad(path pr_excel_prova,path obj_sta_matr, string name){
	vector<string> vector_string;
	vector<string> v_sense,v_entity,v_event,v_action, v_output;// per memorizzare i valori come stringhe
	// to test the file can be open or not
	ifstream excel_data;
	excel_data.open(absolute(pr_excel_prova));
	if(excel_data.fail()){
		cerr<<"verArg_ad:\t"<<UN_OP_FI << absolute(pr_excel_prova)<< endl;
		//return 1;
	}

	while (excel_data.peek()!=EOF)
	{
		string data,sense,entity,event,action,output;
		getline(excel_data,sense,',');
		getline(excel_data,entity,',');
		getline(excel_data,event,',');
		getline(excel_data,action,',');
		getline(excel_data,output,'\n');
		//vector_string.push_back(data);
		/*cout<<"sense:"<< sense<<endl;
		cout<<"entity:"<< entity<<endl;
		cout<<"event:"<< event<<endl;
		cout<<"action:"<< action <<endl;
		cout<<"output:"<< output <<endl;
		//cout<<data<<endl;*/
		v_sense.push_back(sense);
		v_entity.push_back(entity);
		v_event.push_back(event);
		v_action.push_back(action);
		v_output.push_back(output);
	}
	/*	for(int i =0;i <v_sense.size();i++)
	{
		v_sense_int.push_back(stoi(v_sense[i].c_str()));
		v_entity_int.push_back(stoi(v_entity[i].c_str()));
		v_event_int.push_back(stoi(v_event[i].c_str()));
		v_action_int.push_back(stoi(v_action[i].c_str()));
		v_output_int.push_back(stoi(v_output[i].c_str()));
	}
	*/
	excel_data.close();

	//vector<int> v_sense_int, v_entity_int,v_action_int, v_output_int,v_event_int; // pper memorizzare i valori come numeri interi	
	stringstream sslk;
	int index = INDEX_SEEAO; // la prima riga è quella dei momi etc...
	sslk<<"const int arg_ad"<<name<<"[nr_argAd]={";
	vector_string.push_back(sslk.str());
	sslk.str("");
	sslk<<(int) (v_sense.size()-INDEX_SEEAO);// perchè ho che la prima riga del file csv che è un intestazione 
	vector_string.push_back(sslk.str() + ",//nr_seeao");
	sslk.str("");
	vector_string.push_back(max(v_sense,index)+",//n_sense");
	vector_string.push_back(max(v_entity,index)+",//n_entity");
	vector_string.push_back(max(v_event,index)+",//n_event");
	vector_string.push_back(max(v_action,index)+",//n_action");
	vector_string.push_back(max(v_output,index)+",//n_output");
	sslk<<(int) (csv_rows2col(obj_sta_matr)- INDEX_OBJ);
	vector_string.push_back(sslk.str() + "//nr_obj");
	sslk.str("");
	vector_string.push_back("};");
	//ora definisco il numero delle righe dell'obj_matrix

	return vector_string;
}

int csv_rows2col(path obj_sta_matr){
	vector<string> v_sense, v_output;
	ifstream excel_data;
	excel_data.open(absolute(obj_sta_matr));
	if(excel_data.fail()){
		cerr<<"csv_rows2col:\t"<<UN_OP_FI << absolute(obj_sta_matr)<< endl;
		//return 1;
	}


	while (excel_data.peek()!=EOF)
	{
		string data,sense,entity,event,action,output;
		getline(excel_data,sense,',');
		getline(excel_data,output,'\n');
		//vector_string.push_back(data);
		/*cout<<"sense:"<< sense<<endl;
		cout<<"entity:"<< entity<<endl;
		cout<<"event:"<< event<<endl;
		cout<<"action:"<< action <<endl;
		cout<<"output:"<< output <<endl;
		//cout<<data<<endl;*/
		v_sense.push_back(sense);
		v_output.push_back(output);
	}
	/*	for(int i =0;i <v_sense.size();i++)
	{
		v_sense_int.push_back(stoi(v_sense[i].c_str()));
		v_entity_int.push_back(stoi(v_entity[i].c_str()));
		v_event_int.push_back(stoi(v_event[i].c_str()));
		v_action_int.push_back(stoi(v_action[i].c_str()));
		v_output_int.push_back(stoi(v_output[i].c_str()));
	}
	*/
	excel_data.close();

	return v_sense.size();
}


string max (vector<string> v_sense, int index){
		int max=-1; 
		stringstream ss;
		string max_string; 
	for(int i = index; i< v_sense.size(); i++){ 
		int sense_int_temp;
		stringstream sTmp_sense; 
		sTmp_sense<<v_sense[i].c_str();
		sTmp_sense>>sense_int_temp;
		//v_sense_int.push_back(sense_int_temp);
		if(sense_int_temp>max){
			max = sense_int_temp;
		}

	}

	ss<<max;
	ss>>max_string;
	return max_string;
}

int max_int (vector<int> v_sense, int index){
		int max=-1;  
	for(int i = index; i< v_sense.size(); i++){ 
		//v_sense_int.push_back(sense_int_temp);
		if(v_sense[i]>max){
			max = v_sense[i];
		}

	}
	return max;
}



void write_to_file(vector<string> vector_string, path pa_txt){
	//exception handling
  try {
	if(OUT){
    cout << "\nWriting  array contents to file...";
	}

    //open file for writing
   // ofstream fw("C:\\Users\\ribol\\OneDrive - Politecnico di Milano\\Desktop\\documenti\\gabriele\\universit�\\TesiMagistrale\\codice\\stampa_matrici_da_excel\\matrici_prova.txt", std::ofstream::out);
    ofstream fw(pa_txt, std::ofstream::out);
	if(fw.fail()){
		cerr<<"write_to_file:\t"<<UN_OP_FI << pa_txt << endl;
		//return 1;
	}
	if (fw.is_open())
    {
      //store array contents to text file
      for (int i = 0; i < vector_string.size(); i++) {
        fw << vector_string[i] << "\n";
      }
      fw.close();
    }
    else cout << "Problem with opening file";

  }
  catch (const char* msg) {
    cerr << msg << endl;
  }


  if(OUT){
	cout << "\nDone!";
  	cout << "\nPress any key to exit..."<<endl;
  	getchar();
  }
}



vector<string> f_selection_string_vecArg(path pr_csv){
path pa_csv = absolute(pr_csv);
ifstream excel_data;
	excel_data.open(pa_csv);
	// modello stampa
	// const int n_columns = 5; // one for each element 
   //const int n_rows = 5;
	//const int SEEAO[n_rows][n_columns]= {{1,1,1,1,1},{2,1,2,2,2},{3,3,3,3,3},{4,4,4,4,4},{5,5,5,5,4}};
	
	vector<string> vector_string;
	vector<string> v_column,v_value,v_name,v_weight;// per memorizzare i valori come stringhe
	// to test the file can be open or not
	if(excel_data.fail()){
		cerr<<"f_selection_string_vecArg:\t"<<UN_OP_FI << pa_csv << endl;
		//return 1;
	}

	while (excel_data.peek()!=EOF)
	{
		string column,value,name,weight;

		getline(excel_data,column,',');
		getline(excel_data,value,'\n');
		//vector_string.push_back(data);
		/*cout<<"sense:"<< sense<<endl;
		cout<<"entity:"<< entity<<endl;
		cout<<"event:"<< event<<endl;
		cout<<"action:"<< action <<endl;
		cout<<"output:"<< output <<endl;*/
		//cout<<data<<endl;
		v_column.push_back(column);
		v_value.push_back(value);
		
	}
	
	excel_data.close();

	
	// uso dei dati per formato stampa 
	int arraySizeSense = v_column.size();
	//cout << "array size= " << arraySizeSense << endl;
	//vector<string> vector_stampa;
	stringstream ss1;
	int index_init = 1; // togliere la prima riga perchè rappresentano i nomi delle colonne
	
	for(int i = index_init; i < arraySizeSense-1;i ++){ 
		stringstream ss;
		ss << v_value[i]<<","<<"//"<<v_column[i]<<" ["<<i-1<<"]"; 

		vector_string.push_back(ss.str());
		

	} ////////////////////////////////////////////////////////////////////////
	int i = arraySizeSense-1;
	stringstream ss;
		ss << v_value[i]<<"//"<<v_column[i]<<" ["<<i-1<<"]"<<"\n};"; 

		vector_string.push_back(ss.str());

return vector_string;
}




vector<string> f_vecArg_txt(path argv,string name){

	vector<string> vector_string;
	stringstream ssl; 
	//path pr_txt = "txt_vecArg.txt";
	//path pa_txt = absolute(pr_txt);
	//path pr_csv_entity = argv;//"csv_arg.csv";
	//path pr_csv_sense = "csv_selection_sense_v1.csv";


	vector<string> sense= f_selection_string_vecArg(argv);
	ssl<<"//vecArg";
	vector_string.push_back(ssl.str());
	ssl.str("");
	ssl <<"const double arg_us"<<name.c_str()<<"[n_arg_all] = {";
	vector_string.push_back(ssl.str());
	ssl.str("");
	vector_string = add_string(vector_string,sense);		

	return vector_string;
}

vector<string> ad_GloDec_sta(path pr_txt){
	//cout <<"ci sto lavorando\n";
	// read the csv file
	ifstream excel_data;
	excel_data.open(absolute(pr_txt));
	// modello stampa
		//const int caso_obj = ...;
	
	vector<string> vector_string;
	
	vector<string> v_column,v_value;// per memorizzare i valori come stringhe
	// to test the file can be open or not
	if(excel_data.fail()){
		cerr<<"ad_GloDec_sta:\t"<<UN_OP_FI <<absolute(pr_txt)<< endl;
		//return 1;
	}

	while (excel_data.peek()!=EOF)
	{
		string column,value;
		getline(excel_data,column,',');
		getline(excel_data,value,'\n');
		v_column.push_back(column);
		v_value.push_back(value);
	}
	

	excel_data.close();

	
	
  
  // enter location ////////////////////////////////////////////////////

	//int location_length = 10;
	int arraySizeSense = v_column.size();
	//int n_rows =  arraySize/n_columns;
	//cout<<n_rows<<endl;
	//cout<<10%n_columns<<endl;
	//cout << "array size= " << arraySizeSense << endl;
	//vector<string> vector_stampa;
	int index_init = 0;
	// index di partenza, parto da 0 perchè uso tutte le stampe per non considerare la prima riga che ci sono i max
	// e la seconda i nomi delle colonne
	// rapprensenta anche le righe da non contare nella matrice finale
	
	
	for(int i = index_init; i < arraySizeSense;i ++){ 
		stringstream ss;
		ss << v_column[i] <<"=" << v_value[i]<<";" ;

		vector_string.push_back(ss.str());

	} ////////////////////////////////////////////////////////////////////////
	
	return vector_string;
}


vector<int> ad_globDec_dyn(){
	vector<int> vector_string;
	
	/*
	da usare per ottenere (in particolare i numeri per quella stampa userò un' altra funzione associata per far quella stampa)
	const int nr_seeao_max= 58;
	const int nr_impEntity_max = 11;
	const int nr_impSense_max = 5;
	const int nr_obj_max = 7;
	nelle global declaration ed avere i vettori uguali a quelli inseriti e poi zero nelle system declaration
	*/

	// aprire i vari file come faccio nella funzione Dpa_argAll_sysDec_main() : 
	
	// read the csv file
	ifstream excel_data;
	excel_data.open(absolute(command_argv));
	
	// vettore dove memorizzo la parte cha andrò a scrivere
	//vector<string> vector_string;
	
	vector<string> v_type,v_name,v_csv;// per memorizzare i valori come stringhe

	// to test the file can be open or not
	if(excel_data.fail()){
		cerr<<"ad_globDec_dyn:\t"<<UN_OP_FI << absolute(command_argv)<< endl;
		//return 1;
	}
	// read the file considering the data in that csv file 
	while (excel_data.peek()!=EOF)
	{
		string name,type,csv;
		getline(excel_data,type,',');
		getline(excel_data,name,',');
		getline(excel_data,csv,'\n');
		v_type.push_back(type);
		v_name.push_back(name);
		v_csv.push_back(csv);
	}
	// chiudere il file 
	excel_data.close();
	
	//initializzo i vettori per fare in modo di avere quelli dove che mi servono per definire il max 
	vector<int> max_nr_seeao, max_nr_impEntity, max_nr_impSense, max_nr_Obj;

	//max_int (vector<string> v_sense, int index);
	for(int i = INDEX_GEN; i < v_type.size(); i++){
		vector<string> arg_dyn =  vectorStringFromFile(MISSION_FOLDER+BB+ v_name[i]+ BB + v_csv[i]);
		/* ricordando : 
		argv[0],0
		argv[1],HF20.xml
		argv[9],csv_selection_entity_v1.csv
		argv[10],csv_selection_sense_v1.csv
		argv[11],csv_env1_SEEAO_v3.csv
		argv[12],csv_env1_obj_xye.csv
		argv[13],csv_arg.csv
		*/
		max_nr_impEntity.push_back(csv_rows2col(MISSION_FOLDER+BB+ v_name[i]+ BB  + arg_dyn[IDX_CONFIG_ENTITY]));//per entity
		max_nr_impSense.push_back(csv_rows2col(MISSION_FOLDER+BB+ v_name[i]+ BB  + arg_dyn[IDX_CONFIG_SENSE]));//per sense
		max_nr_seeao.push_back(csv_n_rows5col(MISSION_FOLDER+BB+ v_name[i]+ BB  + arg_dyn[IDX_CONFIG_SEEAO])); // per seeao
		max_nr_Obj.push_back(csv_rows2col(MISSION_FOLDER+BB+ v_name[i]+ BB  + arg_dyn[IDX_CONFIG_OBJ])); /// per obj

	}

	//prendere il valore massimo tra quello che ho
	//stampare il valore massimo corrispettivo
	stringstream ss; 

	vector_string.push_back(max_int(max_nr_seeao,0)-INDEX_SEEAO); //[0]

	vector_string.push_back(max_int(max_nr_impEntity,0)-INDEX_IN_IMP_ENTITY);//[1]
		
	vector_string.push_back(max_int(max_nr_impSense,0)-INDEX_IN_IMP_SENSE);//[2]

	vector_string.push_back(max_int(max_nr_Obj,0)-INDEX_OBJ);//[3]

	return vector_string;
}


vector<string> ad_globDec_dyn_format(){
	vector<string> vector_string;
	stringstream ss; 

	ss<<"const int "<<NR_SEEAO<< " = "<<argv_ad_globDec_dyn[GLODEC_INX_SEEAO]<<";";
	vector_string.push_back(ss.str());
	ss.str("");

	ss<<"const int "<<NR_IMP_ENTITY<< " = "<<argv_ad_globDec_dyn[GLODEC_INX_ENTITY]<<";";
	vector_string.push_back(ss.str());
	ss.str("");

	ss<<"const int "<<NR_IMP_SENSE<< " = "<<argv_ad_globDec_dyn[GLODEC_INX_SENSE]<<";";
	vector_string.push_back(ss.str());
	ss.str("");

	ss<<"const int "<<NR_OBJ<< " = "<<argv_ad_globDec_dyn[GLODEC_INX_OBJ]<<";";
	vector_string.push_back(ss.str());
	ss.str("");
	return vector_string;
}


int csv_n_rows5col(path pr_excel_prova){
	ifstream excel_data;
	excel_data.open(absolute(pr_excel_prova));
	// modello stampa
	// const int n_columns = 5; // one for each element 
   //const int n_rows = 5;
	//const int SEEAO[n_rows][n_columns]= {{1,1,1,1,1},{2,1,2,2,2},{3,3,3,3,3},{4,4,4,4,4},{5,5,5,5,4}};
	vector<string> v_sense,v_entity,v_event,v_action, v_output;// per memorizzare i valori come stringhe
	// to test the file can be open or not
	if(excel_data.fail()){
		cerr<<"csv_n_rows5col:\t"<<UN_OP_FI << absolute(pr_excel_prova)<< endl;
		//return 1;
	}

	while (excel_data.peek()!=EOF)
	{
		string data,sense,entity,event,action,output;
		getline(excel_data,sense,',');
		getline(excel_data,entity,',');
		getline(excel_data,event,',');
		getline(excel_data,action,',');
		getline(excel_data,output,'\n');
		v_sense.push_back(sense);
		v_entity.push_back(entity);
		v_event.push_back(event);
		v_action.push_back(action);
		v_output.push_back(output);
	}
	
	excel_data.close();

	

	return v_sense.size();
}





vector<string> matrix_obj_xye_argAll(path pr_csv_obj_xye,string name){
	//cout <<"ci sto lavorando\n";
	// read the csv file
	ifstream excel_data;
	excel_data.open(absolute(pr_csv_obj_xye));
	// modello stampa
	// const int nr_obj = 3;
   	// const int nc_obj = 3;
	//double obj_xye[nr_obj][nc_obj] = {{1.0,1.0,1.0},{200.0,-500.54,2.0},{300.89,-700.5,4.0}};
	vector<string> vector_string;
	string initialization = "const double obj_xye"+ name + "["+NR_OBJ+"][nc_obj] = {";
	string end = "};" ;
	
	vector<string> v_x,v_entity,v_y;
	// to test the file can be open or not
	if(excel_data.fail()){
		cerr<<"matrix_obj_xye_argAll:\t"<<UN_OP_FI << absolute(pr_csv_obj_xye) << endl;
		//return 1;
	}

	while (excel_data.peek()!=EOF)
	{
		string x,y,entity;
		getline(excel_data,x,',');
		getline(excel_data,y,',');
		getline(excel_data,entity,'\n');
		//vector_string.push_back(data);
		/*cout<<"sense:"<< sense<<endl;
		cout<<"entity:"<< entity<<endl;
		cout<<"event:"<< event<<endl;
		cout<<"action:"<< action <<endl;
		cout<<"output:"<< output <<endl;
		//cout<<data<<endl;*/
		v_x.push_back(x);
		v_y.push_back(y);
		v_entity.push_back(entity);

	}
	
	excel_data.close();

	
	
  
  // enter location ////////////////////////////////////////////////////
	int n_columns = 3;
	//int location_length = 10;
	int arraySizeSense = v_x.size();
	//int n_rows =  arraySize/n_columns;
	//cout<<n_rows<<endl;
	//cout<<10%n_columns<<endl;
	if(OUT){
	cout << "array size= " << arraySizeSense << endl;
	}
	//vector<string> vector_stampa;
	stringstream ss;
	//ss<<"const int nr_obj"<<name.c_str()<<" = "<<arraySizeSense-1<<";";
	//vector_string.push_back(ss.str());
	//ss.str("");
	//ss<<"const int nc_obj"<<name.c_str()<<" = "<<3<<";";
	//vector_string.push_back(ss.str());

	vector_string.push_back(initialization);

	for(int i = INDEX_OBJ; i < arraySizeSense-INDEX_OBJ;i ++){ //per evitare di stampare la prima riga che sono i nomi delle colonne
		stringstream ss;
		ss << "{" << v_x[i] <<"," << v_y[i]<<","<<v_entity[i]<<"},"  ;

		vector_string.push_back(ss.str());
		

	} ////////////////////////////////////////////////////////////////////////
	
	// aggiunto perchè l'ultimo ha un stringa diversa, non ha la virgola 
	ss.str("");

	// la sistemo per il caso argAll
	// aggiunto perchè l'ultimo ha un stringa diversa, non ha la virgola 
	//stringstream ss;
	// aggiungere la parte per aggiungere righe che sono pari a zero quando n_rows < nr_seeao_max
	if((v_x.size()-INDEX_OBJ) < argv_ad_globDec_dyn[GLODEC_INX_OBJ] ){
		int i = v_x.size()-INDEX_OBJ ;
		ss << "{" << v_x[i] <<"," << v_y[i]<<","<<v_entity[i]<<"},"  ;
		vector_string.push_back(ss.str());
		ss.str("");

		for(int j = (v_x.size()-INDEX_OBJ); j < argv_ad_globDec_dyn[3]-1; j++){
			ss << SET_0_OBJ <<",";
			vector_string.push_back(ss.str());
			ss.str("");
		}	
			ss << SET_0_OBJ;
			vector_string.push_back(ss.str());
			ss.str("");

	}
	else{
		int i = (v_x.size()-INDEX_OBJ);
		ss << "{" << v_x[i] <<"," << v_y[i]<<","<<v_entity[i]<<"}"  ;
		vector_string.push_back(ss.str());
		ss.str("");
	}


	vector_string.push_back(end);//fine stampa della matrice
  	
	return vector_string;
}




vector<string> f_importance_matrix_num_txt_argAll(path pr_csv_entity,path pr_csv_sense,string name){

	vector<string> vector_string;
	stringstream ssl; 
	//path pr_txt = "txt_importance_matrix_num.txt";
	//path pa_txt = absolute(pr_txt);
	//path pr_csv_entity = "csv_selection_entity_v1.csv";
	//path pr_csv_sense = "csv_selection_sense_v1.csv";


	vector<string> sense= f_selection_string_v2_num_argAll(pr_csv_sense,GLODEC_INX_SENSE,INDEX_IN_IMP_SENSE,SET_0_IMPSENSE);
	ssl<<"//sense importance matrix";
	vector_string.push_back(ssl.str());
	ssl.str("");
	//ssl<<"const int n_sense_tot = 5;";
	//vector_string.push_back(ssl.str());
	ssl.str("");
	ssl <<"const double importance_sense"<<name.c_str()<<"["<<NR_IMP_SENSE<<"] = {";
	vector_string.push_back(ssl.str());
	ssl.str("");
	vector_string = add_string(vector_string,sense);

	vector<string> entity= f_selection_string_v2_num_argAll(pr_csv_entity,GLODEC_INX_ENTITY,INDEX_IN_IMP_ENTITY,SET_0_IMPENTITY);
	ssl<<"//entity importance matrix";
	vector_string.push_back(ssl.str());
	ssl.str("");
	ssl <<"const double importance_entity"<<name.c_str()<<"["<<NR_IMP_ENTITY<<"] = {";
	vector_string.push_back(ssl.str());
	ssl.str("");
	vector_string = add_string(vector_string,entity);
	
	return vector_string;
	
}


vector<string> f_selection_string_v2_num_argAll(path pr_csv,int gloDecInx,int INDEX, string SET_0){
path pa_csv = absolute(pr_csv);
ifstream excel_data;
	excel_data.open(pa_csv);
	// modello stampa
	// const int n_columns = 5; // one for each element 
   //const int n_rows = 5;
	//const int SEEAO[n_rows][n_columns]= {{1,1,1,1,1},{2,1,2,2,2},{3,3,3,3,3},{4,4,4,4,4},{5,5,5,5,4}};
	
	vector<string> vector_string;
	vector<string> v_column,v_value,v_name,v_weight;// per memorizzare i valori come stringhe
	// to test the file can be open or not
	if(excel_data.fail()){
		cerr<<"f_selection_string_v2_num_argAll:\t"<<UN_OP_FI << pa_csv << endl;
		//return 1;
	}

	while (excel_data.peek()!=EOF)
	{
		string column,value,name,weight;

		getline(excel_data,column,',');
		getline(excel_data,value,',');
		getline(excel_data,name,',');
        getline(excel_data,weight,'\n');
		//vector_string.push_back(data);
		/*cout<<"sense:"<< sense<<endl;
		cout<<"entity:"<< entity<<endl;
		cout<<"event:"<< event<<endl;
		cout<<"action:"<< action <<endl;
		cout<<"output:"<< output <<endl;*/
		//cout<<data<<endl;
		v_column.push_back(column);
		v_value.push_back(value);
		v_name.push_back(name);
        v_weight.push_back(weight);
		
	}
	
	excel_data.close();

	
	// uso dei dati per formato stampa 
	//int arraySizeSense = v_column.size();
	if(OUT){
	cout << "array size= " <<v_column.size() << endl;
	}
	//vector<string> vector_stampa;
	//int index_init = 1; // togliere la prima riga perchè rappresentano i nomi delle colonne
	
	// aggiunto perchè l'ultimo ha un stringa diversa, non ha la virgola 
	stringstream ss;

	for(int i = INDEX; i < v_column.size()-INDEX;i ++){ 
		ss << v_weight[i]<<","; 
		vector_string.push_back(ss.str());
		ss.str("");
	} ////////////////////////////////////////////////////////////////////////


	// aggiungere la parte per aggiungere righe che sono pari a zero quando n_rows < nr_seeao_max
	if((v_column.size()-INDEX) < argv_ad_globDec_dyn[gloDecInx] ){
		int i = v_column.size()-INDEX ;
		ss << v_weight[i]<<","  ;
		vector_string.push_back(ss.str());
		ss.str("");

		for(int j = (v_column.size()-INDEX); j < argv_ad_globDec_dyn[gloDecInx]-1; j++){
			ss <<SET_0 <<",";
			vector_string.push_back(ss.str());
			ss.str("");
		}	
			ss << SET_0;
			vector_string.push_back(ss.str());
			ss.str("");

	}
	else{

		int i = (v_weight.size()-INDEX) ;
		ss << v_weight[i]; 
		vector_string.push_back(ss.str());
		ss.str("");
	}


	vector_string.push_back("};");//fine stampa della matrice

return vector_string;
}