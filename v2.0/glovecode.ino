
/**
 * @brief
 * measumrent of globes
 * roughly ~740Hz
 *
 */

int fingers[5] = {A0, A1, A2, A3,A4}; //setup measurement pins
int measurement;
int delay_ms = 0; //delay amount between measurements

void setup()
{
  Serial.begin(9600);
}

void loop()
{

  for(int i = 0; i < 5; i++){
    measurement = analogRead(fingers[i]);
    Serial.print(measurement);
    Serial.print(" ");
  }
    Serial.println(); // Print a newline to separate each set of values
    delayMicroseconds(delay_ms);
}
