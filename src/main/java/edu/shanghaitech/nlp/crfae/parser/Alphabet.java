package edu.shanghaitech.nlp.crfae.parser;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;


public class Alphabet implements Serializable {
    private Map<String, Integer> map;
    private Map<Integer, String> reverseMap;
    private int numEntries;
    private boolean growthStopped = false;

    public Alphabet(int capacity) {
        this.map = new HashMap<>(capacity);
        this.reverseMap = new HashMap<>(capacity);
        numEntries = 0;
    }

    public Alphabet() {
        this(100000);
    }

    /**
     * @param entry
     * @param addIfNotPresent
     * @return The index of `entry` in the table, return -1 if not exists.
     */
    public int lookupIndex(String entry, boolean addIfNotPresent) {
        if (entry == null) {
            throw new IllegalArgumentException("Can't lookup \"null\" in an Alphabet.");
        }

        int ret = map.getOrDefault(entry, -1);

        if (ret == -1 && !growthStopped && addIfNotPresent) {
            ret = numEntries;
            map.put(entry, ret);
            reverseMap.put(ret, entry);
            numEntries++;
        }
        return ret;
    }

    public int lookupIndex(String entry) {
        return lookupIndex(entry, true);
    }

    public String getKeyByIndex(Integer idx) {
        return reverseMap.getOrDefault(idx, "NULL");
    }

    public int size() {
        return numEntries;
    }

    public void stopGrowth() {
        growthStopped = true;
    }

    // Serialization
    private void writeObject(ObjectOutputStream out) throws IOException {
        out.writeInt(numEntries);
        out.writeObject(map);
        out.writeBoolean(growthStopped);
    }

    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        numEntries = in.readInt();
        map = (HashMap<String, Integer>) in.readObject();
        growthStopped = in.readBoolean();
    }
}
