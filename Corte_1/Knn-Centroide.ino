// Opcion A: mide tiempo TOTAL (lectura sensor + clasificacion) al presionar C
// TCS34725 + IQR + EEPROM + KNN manual (optimizada, sin comando A)
// - S: toma 20 lecturas + IQR y acumula sumas por clase
// - W: fuerza cálculo de centroides y escritura en EEPROM
// - P: muestra centroides (EEPROM) y uso EEPROM + contadores
// - C: toma 1 lectura y clasifica usando centroides almacenados (imprime probabilidades)
// Además: mide tiempo total (lectura + clasificacion) en ms.

#include <Wire.h>
#include <Adafruit_TCS34725.h>
#include <EEPROM.h>

Adafruit_TCS34725 tcs(TCS34725_INTEGRATIONTIME_50MS, TCS34725_GAIN_4X);

const uint8_t SAMPLES = 20;           // lecturas por muestreo IQR
const int ADDR_RED = 0, ADDR_ORANGE = 8, ADDR_GREEN = 16, ADDR_MAGIC = 24;
const uint8_t MAGIC_BYTE = 0xA5;

struct ColorVals { uint16_t r,g,b,c; };
ColorVals stored[3]; // centroides leídos desde EEPROM
const char* namesStored[3] = {"ROJO","NARANJA","VERDE"};
const int colorAddrs[3] = { ADDR_RED, ADDR_ORANGE, ADDR_GREEN };

int K = 1;
float datasetX[3][3];
int datasetLabels[3];
int datasetSize = 0;
bool knnTrained = false;

// ---------- optimizaciones de memoria ----------
static uint16_t rS[SAMPLES], gS[SAMPLES], bS[SAMPLES], cS[SAMPLES];
// tmp para ordenar (reutilizable)
static uint16_t tmpArr[SAMPLES];

// sumas acumuladas para centroides (4 sumas por clase) + contador
uint32_t sumR[3] = {0,0,0};
uint32_t sumG[3] = {0,0,0};
uint32_t sumB[3] = {0,0,0};
uint32_t sumC[3] = {0,0,0};
uint16_t cnt[3]   = {0,0,0};
const uint16_t MIN_SAMPLES_PER_CLASS = 3; // ajuste si quieres más muestras antes de auto-escribir
const uint16_t MAX_CNT_LIMIT = 2000; // evita overflow acumulativo (seguro alto)

// ---------- EEPROM helpers ----------
void writeU16(int a,uint16_t v){ EEPROM.update(a, v & 0xFF); EEPROM.update(a+1, v>>8); }
uint16_t readU16(int a){ return (uint16_t)EEPROM.read(a) | ((uint16_t)EEPROM.read(a+1) << 8); }
void setMagic(){ EEPROM.update(ADDR_MAGIC, MAGIC_BYTE); }
bool isMagic(){ return EEPROM.read(ADDR_MAGIC) == MAGIC_BYTE; }

int eepromTotal(){ return EEPROM.length(); }
int eepromUsed(){
  int used=0, total=EEPROM.length();
  for(int i=0;i<total;i++) if(EEPROM.read(i) != 0xFF) used++;
  return used;
}
void printEEPROMUsage(){
  int totalB=eepromTotal(), usedB=eepromUsed();
  long totalBits = (long)totalB*8;
  long usedBits  = (long)usedB*8;
  float pct = (totalB==0)?0.0f:((float)usedB/(float)totalB)*100.0f;
  Serial.print("EEPROM: "); Serial.print(usedB); Serial.print(" bytes (");
  Serial.print(usedBits); Serial.print(" bits) de ");
  Serial.print(totalB); Serial.print(" bytes (");
  Serial.print(totalBits); Serial.print(" bits) -> ");
  Serial.print(pct,2); Serial.println("% usados.");
}

// ---------- util: insertion sort (in-place) ----------
void insertion_sort(uint16_t a[], uint8_t n){
  for(uint8_t i=1;i<n;i++){
    uint16_t key = a[i];
    int j = i-1;
    while(j>=0 && a[j] > key){
      a[j+1] = a[j];
      j--;
      if(j<0) break;
    }
    a[j+1] = key;
  }
}

// ---------- IQR cleaning (entero) ----------
uint16_t cleanIQR_uint(uint16_t samples[], uint8_t n){
  if(n==0) return 0;
  for(uint8_t i=0;i<n;i++) tmpArr[i] = samples[i];
  insertion_sort(tmpArr, n);
  int32_t q1, q3;
  if(n%2==0){
    int mid = n/2;
    if(mid%2==1){ q1 = tmpArr[mid/2]; }
    else { q1 = ( (int32_t)tmpArr[mid/2 - 1] + (int32_t)tmpArr[mid/2] ) / 2; }
    int upLen = n - mid;
    if(upLen%2==1) q3 = tmpArr[mid + upLen/2];
    else { q3 = ( (int32_t)tmpArr[mid + upLen/2 -1] + (int32_t)tmpArr[mid + upLen/2] ) / 2; }
  } else {
    int mid = n/2;
    int lowLen = mid;
    if(lowLen==0) q1 = tmpArr[0];
    else if(lowLen%2==1) q1 = tmpArr[lowLen/2];
    else q1 = ( (int32_t)tmpArr[lowLen/2 -1] + (int32_t)tmpArr[lowLen/2] ) / 2;
    int upLen = n - mid -1;
    if(upLen==0) q3 = tmpArr[mid];
    else if(upLen%2==1) q3 = tmpArr[mid + 1 + upLen/2];
    else q3 = ( (int32_t)tmpArr[mid + 1 + upLen/2 -1] + (int32_t)tmpArr[mid + 1 + upLen/2] ) / 2;
  }
  int32_t iqr = q3 - q1;
  int32_t low  = q1 - (3*iqr)/2;
  int32_t high = q3 + (3*iqr)/2;
  uint32_t sum = 0; uint16_t kept = 0;
  for(uint8_t i=0;i<n;i++){
    int32_t v = (int32_t)samples[i];
    if(v >= low && v <= high){ sum += (uint32_t)v; kept++; }
  }
  if(kept == 0){
    if(n%2) return tmpArr[n/2];
    else return (uint16_t)(((uint32_t)tmpArr[n/2 -1] + (uint32_t)tmpArr[n/2]) / 2);
  }
  uint32_t avg = (sum + kept/2) / kept;
  if(avg > 65535) avg = 65535;
  return (uint16_t)avg;
}

// ---------- muestreo y IQR ----------
void takeSampleIQR(uint16_t &rOut,uint16_t &gOut,uint16_t &bOut,uint16_t &cOut, bool verbose=true){
  for(uint8_t i=0;i<SAMPLES;i++){
    uint16_t r,g,b,c;
    tcs.getRawData(&r,&g,&b,&c);
    rS[i]=r; gS[i]=g; bS[i]=b; cS[i]=c;
    if(verbose){
      Serial.print("muestra "); Serial.print(i+1);
      Serial.print(" R:"); Serial.print(r);
      Serial.print(" G:"); Serial.print(g);
      Serial.print(" B:"); Serial.print(b);
      Serial.print(" C:"); Serial.println(c);
    }
    delay(120);
  }
  rOut = cleanIQR_uint(rS, SAMPLES);
  gOut = cleanIQR_uint(gS, SAMPLES);
  bOut = cleanIQR_uint(bS, SAMPLES);
  cOut = cleanIQR_uint(cS, SAMPLES);
}

// ---------- lectura sencilla para clasificación ----------
void readOne(uint16_t &r,uint16_t &g,uint16_t &b,uint16_t &c, bool verbose=true){
  tcs.getRawData(&r,&g,&b,&c);
  if(verbose){
    Serial.print("one -> R:"); Serial.print(r);
    Serial.print(" G:"); Serial.print(g);
    Serial.print(" B:"); Serial.print(b);
    Serial.print(" C:"); Serial.println(c);
  }
}

// ---------- manejo sumas/contadores ----------
void addToSums(int cls, uint16_t r,uint16_t g,uint16_t b,uint16_t c){
  if(cls<0 || cls>2) return;
  if(cnt[cls] < MAX_CNT_LIMIT){
    sumR[cls] += r; sumG[cls] += g; sumB[cls] += b; sumC[cls] += c; cnt[cls]++;
  } else {
    sumR[cls] = (sumR[cls] >> 1) + r;
    sumG[cls] = (sumG[cls] >> 1) + g;
    sumB[cls] = (sumB[cls] >> 1) + b;
    sumC[cls] = (sumC[cls] >> 1) + c;
    cnt[cls] = (cnt[cls] >> 1) + 1;
  }
  Serial.print("added to "); Serial.print(namesStored[cls]); Serial.print(" cnt="); Serial.println(cnt[cls]);
}

// ---------- calcular centroides y escribir en EEPROM ----------
void computeWriteCentroids(bool force=false){
  if(!force){
    for(int i=0;i<3;i++){
      if(cnt[i] < MIN_SAMPLES_PER_CLASS){
        Serial.print("esperando más muestras para "); Serial.print(namesStored[i]); Serial.print(" (");
        Serial.print(cnt[i]); Serial.print("/"); Serial.print(MIN_SAMPLES_PER_CLASS); Serial.println(")");
        return;
      }
    }
  }
  for(int cls=0; cls<3; cls++){
    if(cnt[cls] == 0){
      writeU16(colorAddrs[cls]+0, 0);
      writeU16(colorAddrs[cls]+2, 0);
      writeU16(colorAddrs[cls]+4, 0);
      writeU16(colorAddrs[cls]+6, 0);
      continue;
    }
    uint16_t cr = (uint16_t)((sumR[cls] + cnt[cls]/2) / cnt[cls]);
    uint16_t cg = (uint16_t)((sumG[cls] + cnt[cls]/2) / cnt[cls]);
    uint16_t cb = (uint16_t)((sumB[cls] + cnt[cls]/2) / cnt[cls]);
    uint16_t cc = (uint16_t)((sumC[cls] + cnt[cls]/2) / cnt[cls]);
    writeU16(colorAddrs[cls]+0, cr);
    writeU16(colorAddrs[cls]+2, cg);
    writeU16(colorAddrs[cls]+4, cb);
    writeU16(colorAddrs[cls]+6, cc);
    Serial.print("centroid escrito "); Serial.print(namesStored[cls]); Serial.print(" R:"); Serial.print(cr);
    Serial.print(" G:"); Serial.print(cg); Serial.print(" B:"); Serial.print(cb); Serial.print(" C:"); Serial.println(cc);
  }
  setMagic();
  for(int i=0;i<3;i++){
    int a=colorAddrs[i];
    stored[i].r = readU16(a+0);
    stored[i].g = readU16(a+2);
    stored[i].b = readU16(a+4);
    stored[i].c = readU16(a+6);
  }
  for(int i=0;i<3;i++){ sumR[i]=sumG[i]=sumB[i]=sumC[i]=0; cnt[i]=0; }
  datasetSize=0;
  for(int i=0;i<3;i++){
    ColorVals sc = stored[i];
    if(sc.r==0 && sc.g==0 && sc.b==0 && sc.c==0) continue;
    float rn=(sc.c>0)?((float)sc.r/(float)sc.c):((sc.r+sc.g+sc.b)>0? (float)sc.r/(float)(sc.r+sc.g+sc.b):0.0f);
    float gn=(sc.c>0)?((float)sc.g/(float)sc.c):((sc.r+sc.g+sc.b)>0? (float)sc.g/(float)(sc.r+sc.g+sc.b):0.0f);
    float bn=(sc.c>0)?((float)sc.b/(float)sc.c):((sc.r+sc.g+sc.b)>0? (float)sc.b/(float)(sc.r+sc.g+sc.b):0.0f);
    datasetX[datasetSize][0]=rn; datasetX[datasetSize][1]=gn; datasetX[datasetSize][2]=bn;
    datasetLabels[datasetSize]=i; datasetSize++;
  }
  knnTrained = datasetSize>0;
  printEEPROMUsage();
}

// ---------- KNN (usa distancia al cuadrado para evitar sqrt) ----------
float dist2(const float a[3], float b0,float b1,float b2){
  float dr=a[0]-b0, dg=a[1]-b1, db=a[2]-b2;
  return dr*dr + dg*dg + db*db;
}

// ---------- classifyManual mejorado: imprime probabilidades ----------
int classifyManual(uint16_t r,uint16_t g,uint16_t b,uint16_t c){
  if(!knnTrained){ Serial.println("KNN no entrenado. Usa S y luego W."); return -1; }
  float rn,gn,bn;
  if(c>0){ rn=(float)r/(float)c; gn=(float)g/(float)c; bn=(float)b/(float)c; }
  else { uint32_t s=(uint32_t)r+g+b; if(s>0){ rn=(float)r/s; gn=(float)g/s; bn=(float)b/s; } else rn=gn=bn=0.0f; }
  float dists[3]; int labels[3];
  const float INF = 1e12f;
  for(int i=0;i<datasetSize;i++){ dists[i] = dist2(datasetX[i], rn,gn,bn); labels[i]=datasetLabels[i]; }
  for(int i=datasetSize;i<3;i++){ dists[i]=INF; labels[i]=-1; }

  int kActual = (K>datasetSize)?datasetSize:K;
  bool taken[3] = {false,false,false};
  int votes[3] = {0,0,0};
  int nearestLabel=-1; float nearestD=INF;
  for(int k=0;k<kActual;k++){
    int best=-1; float bestD=INF;
    for(int i=0;i<datasetSize;i++) if(!taken[i] && dists[i]<bestD){ bestD=dists[i]; best=i; }
    if(best==-1) break;
    taken[best]=true;
    if(labels[best]>=0 && labels[best]<3) votes[labels[best]]++;
    if(bestD < nearestD){ nearestD = bestD; nearestLabel = labels[best]; }
  }
  int bestLabel=-1, bestVotes=-1;
  for(int lbl=0;lbl<3;lbl++) if(votes[lbl]>bestVotes){ bestVotes=votes[lbl]; bestLabel=lbl; }
  if(bestVotes>0){
    int ties=0; for(int lbl=0;lbl<3;lbl++) if(votes[lbl]==bestVotes) ties++;
    if(ties>1 && nearestLabel!=-1) bestLabel = nearestLabel;
  } else bestLabel = nearestLabel;

  Serial.println("--- result ---");
  Serial.print("norm r:"); Serial.print(rn,4); Serial.print(" g:"); Serial.print(gn,4); Serial.print(" b:"); Serial.println(bn,4);
  if(bestLabel>=0) { Serial.print("Clase (voto KNN): "); Serial.println(namesStored[bestLabel]); }
  else Serial.println("Indeterminado (votos)");

  // probabilidades por clase (peso inverso por distancia)
  float classWeights[3] = {0.0f, 0.0f, 0.0f};
  bool hasZero = false;
  for(int i=0;i<datasetSize;i++) if(dists[i] <= 0.0f) hasZero = true;

  if(hasZero){
    int zeroCounts[3] = {0,0,0};
    int totalZeros = 0;
    for(int i=0;i<datasetSize;i++){
      if(dists[i] <= 0.0f && labels[i]>=0 && labels[i]<3){
        zeroCounts[labels[i]]++;
        totalZeros++;
      }
    }
    if(totalZeros==0){ classWeights[0]=classWeights[1]=classWeights[2]=1.0f; }
    else { for(int lbl=0; lbl<3; lbl++) classWeights[lbl] = (float)zeroCounts[lbl]; }
  } else {
    const float EPS = 1e-6f;
    for(int i=0;i<datasetSize;i++){
      if(labels[i]>=0 && labels[i]<3){
        float w = 1.0f / (dists[i] + EPS);
        classWeights[labels[i]] += w;
      }
    }
    if(classWeights[0]==0.0f && classWeights[1]==0.0f && classWeights[2]==0.0f){
      classWeights[0]=classWeights[1]=classWeights[2]=1.0f;
    }
  }

  float totalW = classWeights[0] + classWeights[1] + classWeights[2];
  int perc[3] = {0,0,0};
  if(totalW <= 0.0f){
    perc[0] = perc[1] = perc[2] = 33; perc[2]=34;
  } else {
    int acc = 0;
    for(int i=0;i<3;i++){
      float p = (classWeights[i] / totalW) * 100.0f;
      perc[i] = (int)(p + 0.5f);
      acc += perc[i];
    }
    if(acc != 100){
      int diff = 100 - acc;
      int bestI = 0; float bestW = classWeights[0];
      for(int i=1;i<3;i++) if(classWeights[i] > bestW){ bestW = classWeights[i]; bestI = i; }
      perc[bestI] += diff;
    }
  }

  Serial.print("Probabilidades: La muestra fue ");
  Serial.print(namesStored[0]); Serial.print(" "); Serial.print(perc[0]); Serial.print("%, ");
  Serial.print(namesStored[1]); Serial.print(" "); Serial.print(perc[1]); Serial.print("%, ");
  Serial.print(namesStored[2]); Serial.print(" "); Serial.print(perc[2]); Serial.println("%.");

  return bestLabel;
}

// ---------- scan (S) ----------
void scanStoreOne(int cls){
  Serial.print("scan S -> "); Serial.println(namesStored[cls]);
  uint16_t r,g,b,c;
  takeSampleIQR(r,g,b,c,true);
  addToSums(cls, r,g,b,c);
  bool ok=true; for(int i=0;i<3;i++) if(cnt[i] < MIN_SAMPLES_PER_CLASS) ok=false;
  if(ok){ Serial.println("todos cumplen. calculando centroides..."); computeWriteCentroids(false); }
  else Serial.println("muestra añadida. usa W para forzar escritura.");
  printEEPROMUsage();
}

// ---------- load stored ----------
void loadStored(){
  if(!isMagic()){ for(int i=0;i<3;i++) stored[i]={0,0,0,0}; Serial.println("EEPROM: no valida"); return; }
  for(int i=0;i<3;i++){
    int a=colorAddrs[i];
    stored[i].r = readU16(a+0);
    stored[i].g = readU16(a+2);
    stored[i].b = readU16(a+4);
    stored[i].c = readU16(a+6);
  }
  datasetSize=0;
  for(int i=0;i<3;i++){
    ColorVals sc=stored[i];
    if(sc.r==0 && sc.g==0 && sc.b==0 && sc.c==0) continue;
    float rn=(sc.c>0)?((float)sc.r/(float)sc.c):((sc.r+sc.g+sc.b)>0? (float)sc.r/(float)(sc.r+sc.g+sc.b):0.0f);
    float gn=(sc.c>0)?((float)sc.g/(float)sc.c):((sc.r+sc.g+sc.b)>0? (float)sc.g/(float)(sc.r+sc.g+sc.b):0.0f);
    float bn=(sc.c>0)?((float)sc.b/(float)sc.c):((sc.r+sc.g+sc.b)>0? (float)sc.b/(float)(sc.r+sc.g+sc.b):0.0f);
    datasetX[datasetSize][0]=rn; datasetX[datasetSize][1]=gn; datasetX[datasetSize][2]=bn;
    datasetLabels[datasetSize]=i; datasetSize++;
  }
  knnTrained = datasetSize>0;
}

// ---------- P print ----------
void printAll(){
  Serial.println("--- centroides (EEPROM) ---");
  if(!isMagic()) Serial.println("No calibracion valida");
  for(int i=0;i<3;i++){
    Serial.print(namesStored[i]); Serial.print(" R:"); Serial.print(stored[i].r);
    Serial.print(" G:"); Serial.print(stored[i].g);
    Serial.print(" B:"); Serial.print(stored[i].b);
    Serial.print(" C:"); Serial.println(stored[i].c);
  }
  printEEPROMUsage();
  Serial.println("--- contadores RAM ---");
  for(int i=0;i<3;i++){
    Serial.print(namesStored[i]); Serial.print(" cnt="); Serial.print(cnt[i]);
    Serial.print(" sumR="); Serial.print(sumR[i]); Serial.print(" sumG="); Serial.print(sumG[i]);
    Serial.print(" sumB="); Serial.print(sumB[i]); Serial.print(" sumC="); Serial.println(sumC[i]);
  }
  Serial.println("-----------------------");
}

// ---------- setup / loop ----------
void setup(){
  Serial.begin(115200);
  delay(200);
  Serial.println("TCS34725 + IQR + EEPROM + KNN (Opcion A: tiempo TOTAL)");
  if(!tcs.begin()){ Serial.println("ERROR tcs.begin"); while(1) delay(500); }
  loadStored();
  Serial.println("Comandos: S (guardar siguiente), P (imprimir), C (clasificar 1 lectura), K<n>, W (forzar centroides)");
}

void loop(){
  static char buf[48]; static uint8_t idx=0; static int nextColor = 0;
  while(Serial.available()){
    char ch = (char)Serial.read();
    if(ch=='\r' || ch=='\n'){
      if(idx>0){
        buf[idx]=0; String line=String(buf); line.trim(); idx=0;
        char c = line.charAt(0);
        if(c=='S' || c=='s'){ scanStoreOne(nextColor); nextColor=(nextColor+1)%3; Serial.print("next: "); Serial.println(namesStored[nextColor]); }
        else if(c=='P' || c=='p') printAll();
        else if(c=='C' || c=='c'){
          unsigned long t0 = micros();
          uint16_t r,g,b,cc;
          readOne(r,g,b,cc,true);
          int lbl = classifyManual(r,g,b,cc);
          unsigned long t1 = micros();
          float elapsed_ms = (t1 - t0) / 1000.0f;
          Serial.print("Tiempo total (lectura + clasificacion): ");
          Serial.print(elapsed_ms, 3);
          Serial.println(" ms");
        }
        else if(c=='W' || c=='w'){ Serial.println("Forzando centroides..."); computeWriteCentroids(true); }
        else if(c=='K' || c=='k'){ String num=line.substring(1); num.trim(); if(num.length()>0){ int nk=num.toInt(); if(nk<=0) Serial.println("K>=1"); else { K=nk; Serial.print("K="); Serial.println(K);} } else { Serial.print("K actual="); Serial.println(K);} }
        else Serial.println("Comando no reconocido. Usa S,P,C,K<n>,W.");
      }
    } else { if(idx < sizeof(buf)-1) buf[idx++] = ch; else idx=0; }
  }
  delay(10);
}
