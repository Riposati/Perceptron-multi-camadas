#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void inicializaVetorPesos(double w[201][4]){
    int i,j;
    srand((double) time(NULL)); // mudar a sequencia usando o relógio

    printf("vetor pesos sinapticos original:\n");
    for(i=0;i<201;i++){ // inicializar os pesos c valores aleatórios entre 0 e 1
        for(j=0;j<4;j++){
            w[i][j] = ((double) rand() / (double)(RAND_MAX));
            printf("%lf ",w[i][j]);
        }
        printf("\n");
    }
}

void inicializaEntradas(double entradas[201][4]){

    /* BIAS */
    entradas[0][0] = -1;
    entradas[0][1] = -1;
    entradas[0][2] = -1;
    entradas[0][3] = -1;

    FILE *arq;
    arq = fopen("entradas.txt", "rt");

    if(arq==NULL){
        printf("problema na abertura do arquivo\n");
        return;
    }

    int i,j;
    for(i=1;i<201;i++){
        for(j=0;j<4;j++){
            if (!fscanf(arq, "%lf", &entradas[i][j]))
                break;
            //printf("%lf ",entradas[i][j]);
        }
        //printf("\n");
    }
    fclose(arq);
}

double g(double u){ // funcao logistica
    double e = 2.718281;
    double beta = 0.0005;
    return (1 / (1+ pow(e,-(beta*u))));
}

double derivadaFuncaoLogistica(double u){

    return (0.0005*pow(2.718281,(-0.0005*u))) / (1+pow(2.718281,pow((-0.0005*u),2)));
}

double ErroMedio(double w[201][4], double entradas[201][4]){

    int i,j,z;
    double erro=0.0,l[2],p=200.0,y[2];

    for(z=0;z<2;z++){
        l[z] = 0.0;
        y[z] = 0.0;
    }

    for(z=0;z<2;z++){ // n camadas
        for(i=0;i<4;i++){
            for(j=0;j<201;j++){
                if(i==0){
                    l[z] += w[j][i]*entradas[j][3]; // a saida do primeiro neuronio
                }
                else if(z > 0){
                    int r = z-1;
                    l[z] += w[j][i]*y[r]; // camadas posteriores a primeira
                }
            }
        }

        y[z] = g(l[z]);
    }

    /*printf("y[z] = %lf\n",y[0]);
    printf("y[z] = %lf\n",y[1]);

    printf("l[z] = %lf\n",l[0]);
    printf("l[z] = %lf\n",l[1]);

    printf("g(l[z]) = %lf\n",g(l[0]));
    printf("g(l[z]) = %lf\n",g(l[1]));*/

    for(z=0;z<2;z++){
        for(i=0;i<201;i++){
            erro += pow(entradas[i][3] - y[1],2);
        }
    }

    erro = erro * 0.5;
    double erroMedio = (1 / p)*erro;

    printf("%lf,\n",erroMedio);

    return erroMedio;
}

int main()
{
    // se em n épocas nao convergir a rede para pois o conjunto nao é linearmente separável
    //int barreiraDeSegurancaEpocas = 50000;

    double entradas[201][4];
    inicializaEntradas(entradas); // saidas desejadas estão junto as entradas

    double w[201][4]; // pesos sinápticos
    inicializaVetorPesos(w);

    double taxaAprendizado = 0.1,precisao = 0.000001;
    int contEpocas = 0;
    double eqmAtual=0.0, eqmAnterior=0.0,verifica=0.0;

    int i,j,z,cont=0;
    double v[201],l[2],y[2],gradiente[2];

    /// algoritmo de treinamento backpropagation
    do{
        eqmAnterior = ErroMedio(w,entradas);

        /// fase forward ( produto escalar )
        int i,j,z;
        double erro=0.0,l[2],p=200.0,y[2];

        for(z=0;z<2;z++){
            l[z] = 0.0;
            y[z] = 0.0;
        }

        for(z=0;z<2;z++){ // n camadas
            for(i=0;i<4;i++){
                for(j=0;j<201;j++){
                    if(i==0){
                        l[z] += w[j][i]*entradas[j][3]; // a saida do primeiro neuronio
                    }
                    else if(z > 0){
                        int r = z-1;
                        l[z] += w[j][i]*y[r]; // camadas posteriores a primeira
                    }
                }
            }
            y[z] = g(l[z]);
        }

        /// fase backward

        double deltaCamada2 = 0.0;
        for(i=0;i<201;i++){
            deltaCamada2 += (entradas[i][3]-y[1]) * derivadaFuncaoLogistica(l[0]);
        }
        //printf("delta camada 2 = %lf\n",deltaCamada2);

        for(i=3;i>=0;i--){
            for(j=0;j<201;j++){
                w[j][i] = w[j][i] + taxaAprendizado * deltaCamada2 * y[0];
            }
        }

        double deltaCamada1 = 0.0;
        for(i=3;i>=0;i--){
            for(j=0;j<201;j++){
                deltaCamada1 += ((deltaCamada2*w[j][i]) * derivadaFuncaoLogistica(l[1]));
            }
        }
        deltaCamada1 *=-1;
        for(i=3;i>=0;i--){
            for(j=0;j<201;j++){
                w[j][i] = w[j][i] + taxaAprendizado * deltaCamada1 * y[1];
            }
        }
        //printf("delta camada 1 = %lf\n",deltaCamada1);

        contEpocas++;
        eqmAtual = ErroMedio(w,entradas);

        /*printf("eqmAtual = %lf\n",eqmAtual);
        printf("eqmAnterior = %lf\n",eqmAnterior);*/

    }while((eqmAnterior - eqmAtual) >= precisao);

    printf("vetor pesos sinapticos final:\n");

    for(i=0;i<201;i++){
        for(j=0;j<4;j++){

            printf("%lf ",w[i][j]);
        }
        printf("\n");
    }

    printf("@@@@@@@@@@@@@@@@\n\n");
    printf("qtd epocas = %d\n\n",contEpocas);

    // Fase de operação
    double testes[20][4];

    FILE *arq;
    arq = fopen("saidasDesejadas.txt", "rt");

    if(arq==NULL){
        printf("problema na abertura do arquivo\n");
        return;
    }

    printf("\nvetor com as saidas desejadas:\n");
    for(i=1;i<20;i++){
        for(j=0;j<4;j++){
            if (!fscanf(arq, "%lf", &testes[i][j]))
                break;
            printf("%lf ",testes[i][j]);
        }
        printf("\n");
    }
    fclose(arq);

    for(z=0;z<2;z++){
        l[z] = 0.0;
        y[z] = 0.0;
    }

    int r;
    for(r=0;r<20;r++){
        for(z=0;z<2;z++){ // n camadas
            for(i=0;i<4;i++){
                for(j=0;j<20;j++){
                    if(i==0){
                        l[z] += w[j][i] * testes[j][i]; // a saida do primeiro neuronio
                    }
                    else if(z > 0){
                        int r = z-1;
                        l[z] += w[j][i]*y[r]; // camadas posteriores a primeira
                    }
                }
            }
            y[z] = g(l[z]);
            if(z==1) // valores que o y carrega na ultima camada neural
                printf("y = %lf\n",y[z]);
        }
    }
    /*printf("\n");
    system("pause");*/
    return 0;
}
