import pandas as pd


def tag_entries_with_multiple_labels(excel_path, label_column, label_mappings):
    # Load the Excel file
    df = pd.read_excel(excel_path)

    # Add the label column if it does not exist
    if label_column not in df.columns:
        df[label_column] = None

    # Iterate through the label mappings and update the DataFrame
    for label, entries in label_mappings.items():
        for entry in entries:
            df.loc[df['BADP名称'] == entry, label_column] = label

    # Save the updated DataFrame to the Excel file
    df.to_excel(excel_path, index=False)


if __name__ == '__main__':
    # The path to your Excel file
    excel_file_path = "D:\\demo_exe\\DPBA-MD\\BADP-MD\\BADP-MD\\data\\data_BADP_76.xlsx"  # Replace with the path to your Excel file
    label_col_name = 'label'

    # Mapping of entries to their respective labels
    label_mappings = {
        'label_1': ["Key-value store", "Address mapping", "Blocklist", "Data Contract", "Off-chain data storage",
                 "State Aggregation", "Snapshotting", "Token burning", "Node Sync", "Establish Genesis",
                 "State Initialization", "Exchange Transfer", "Transaction Replay", "Virtual Machine Emulation",
                 "Smart Contract Translation", "Encrypting on-chain data", "Blockchain Anchor", "One-Off Access"],
        'label_2': ["State Channel", "(Off-chain) Contract Registry", "Embedded Permission", "Factory Contract",
                    "Incentive Execution", "Commit and Reveal", "Proxy Contract", "Dynamic Binding",
                    "Flyweight", "Tight Variable Packing", "Legal and smart-contract pair", "Multiple authorization",
                    "Off-chain secret enabled dynamic authentication", "Digital Record", "State machine",
                    "Contract Registry", "Delegated Computation", "Low Contract Footprint", "Contract Composer",
                    "Contract Decorator", "Contract Mediator", "Contract Observer", "Hash Secret"],
        'label_3': ["Oracle", "Bulletin Board", "Announcement", "Migration",
                    "Emergency Stop", "Mutex", "Contract Balance Limit", "Reverse Oracle",
                    "Access Restriction", "Satellite", "Speed Bump", "Rate Limit",
                    "Challenge Response", "Off-chain Signatures"],
        'label_4': ["Master & Sub Key", "Hot & Cold Wallet Storage", "Key Sharding", "Multiple Registration",
                    "Bound with Social Media", "Dual Resolution", "Identifier Registry", "Selective Content Generation",
                    "Time-Constrained Access", "Hard Fork"],
        'label_5': ["Tokenisation", "X-confirmation", "Self-Generated Transactions", "Self-Confirmed Transactions",
                    "Delegated Transactions", "Push-based inbound oracle", "Push-based outbound oracle"],
                    }

    # Call the function
    tag_entries_with_multiple_labels(excel_file_path, label_col_name, label_mappings)
