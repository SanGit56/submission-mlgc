const { Firestore } = require('@google-cloud/firestore');
 
async function storeData(id, data) {
  const db = new Firestore();
  const predictCollection = db.collection("predictions");
 
  try {
    await predictCollection.doc(id).set(data);
    return { success: true, message: 'Data stored successfully' };
  } catch (error) {
    console.error('Error storing data:', error);
    throw new Error('Failed to store data');
  }
}
 
module.exports = storeData;