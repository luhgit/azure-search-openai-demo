import { Example } from "./Example";

import styles from "./Example.module.css";

export type ExampleModel = {
    text: string;
    value: string;
};

const EXAMPLES: ExampleModel[] = [
    {
        text: "What programs are avaiable in a Bosch washing machine?",
        value: "What programs are available in a Bosch washing machine?"
    },
    { text: "How do I enable Wi-Fi on a Siemens dishwasher?", value: "How do I enable Wi-Fi on a Siemens dishwasher?" },
    { text: "How do I add laundry to a Bosch washing machine?", value: "How do I add laundry to a Bosch washing machine?" }
];

interface Props {
    onExampleClicked: (value: string) => void;
}

export const ExampleList = ({ onExampleClicked }: Props) => {
    return (
        <ul className={styles.examplesNavList}>
            {EXAMPLES.map((x, i) => (
                <li key={i}>
                    <Example text={x.text} value={x.value} onClick={onExampleClicked} />
                </li>
            ))}
        </ul>
    );
};
